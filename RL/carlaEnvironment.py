import logging
import random
import numpy as np
import Pyro4
import carla
import gymnasium as gym

from agents.tools.misc import get_speed
from RL.env_utils import build_state_vector
from RL.steering_agent import SteeringAgent

# light logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@Pyro4.expose
class CarlaEnv(gym.Env):
    """A CARLA environment exposed via Pyro.
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        super().__init__()

        self.world = world
        self.ego_vehicle = vehicle

        # Action: single continuous value in [-1, 1], positive == throttle, negative == brake
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: sequence of future waypoint x/y normalized + speed, accel, dist_to_obj_ahead
        self.waypoints_size = 14
        self.lane_width = 3.5
        obs_size = self.waypoints_size * 2 + 3  # x,y per waypoint + speed + accel + dist
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Behavior agent handles steering
        self.agent = SteeringAgent(self.ego_vehicle, behavior='normal')

        # Sync simulation settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        self.world.apply_settings(settings)

        # Radar sensor for forward distance measurement (non-blocking)
        self.max_dist_ahead = 50.0
        self._setup_radar()

        # Episode bookkeeping
        self.episode_step = 0

        # Destination and route metadata (set in reset)
        self.destination = None
        self.max_route_dist = 200.0  # used to normalize potential reward

        logging.info("CarlaEnv initialized")

    def get_action_space(self):
        return self.action_space

    # ---------------------- sensors ----------------------
    def _setup_radar(self):
        try:
            bp = self.world.get_blueprint_library().find('sensor.other.radar')
            bp.set_attribute('horizontal_fov', '30')
            bp.set_attribute('vertical_fov', '5')
            bp.set_attribute('range', '50')
            radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
            self.radar_sensor = self.world.spawn_actor(bp, radar_transform, attach_to=self.ego_vehicle)
            self.radar_sensor.listen(self.radar_callback)
            logging.info("Radar sensor spawned")
        except Exception:
            logging.exception("Failed to spawn radar sensor")
            self.radar_sensor = None

    def radar_callback(self, data):
        # find nearest forward detection (depth attribute on detection)
        try:
            depths = [d.depth for d in data]
            self.max_dist_ahead = float(min(depths)) if depths else 50.0
        except Exception:
            # don't let sensor errors crash the server
            logging.exception("radar callback error")
            self.max_dist_ahead = 50.0

    # ---------------------- gym API ----------------------
    def reset(self, seed=None, options=None):
        # Choose random spawn and destination (destination != spawn)
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        possible_destinations = [p for p in spawn_points if p.location != spawn_point.location]
        destination = random.choice(possible_destinations)

        # place vehicle
        self.ego_vehicle.set_transform(spawn_point)
        # ensure physics updated
        try:
            self.world.tick()
        except Exception:
            logging.exception("world.tick() failed in reset")

        # set destination (store explicitly)
        self.destination = destination.location

        # inform agent (global/local planner)
        try:
            self.agent.set_destination(self.destination)
        except Exception:
            logging.exception("agent.set_destination failed")

        # reset vehicle control and sensors
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        self.max_dist_ahead = 50.0
        self.episode_step = 0

        # initial observation
        obs = self.get_obs()
        return obs.tolist(), {}

    def step(self, action):
        try:
            # --- apply action (throttle/brake) ---
            a = float(np.array(action).squeeze())
            a = float(np.clip(a, -1.0, 1.0))
            throttle = a if a > 0 else 0.0
            brake = -a if a < 0 else 0.0

            # agent steering
            agent_control = self.agent.run_step()
            steer = agent_control.steer

            control = carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer=float(steer))
            self.ego_vehicle.apply_control(control)

            # step simulation
            self.world.tick()
            self.episode_step += 1

            # observations and reward
            current_waypoints = self.agent.get_waypoints()  # fresh local window
            obs = self.get_obs(current_waypoints)
            reward, terminated = self.compute_reward(current_waypoints)

            control_info = {
                "throttle": control.throttle,
                "brake": control.brake,
                "steer": control.steer
            }

            truncated = False
            if self.episode_step > 1000:
                truncated = True



            return obs.tolist(), float(reward), bool(terminated), bool(truncated), control_info

        except Exception as e:
            logging.exception("Error during step()")
            # raise a simple error message for Pyro to transmit
            raise RuntimeError(f"Server step crashed: {e}")

    def close(self):
        try:
            if getattr(self, 'radar_sensor', None):
                self.radar_sensor.stop()
                self.radar_sensor.destroy()
                self.radar_sensor = None
        except Exception:
            logging.exception("Error during close()")

    def render(self):
        pass

    # ---------------------- observations ----------------------
    def get_obs(self, waypoints=None):
        if waypoints is None:
            waypoints = self.agent.get_waypoints()

        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        speed = np.linalg.norm([velocity.x, velocity.y])
        accel = np.linalg.norm([acceleration.x, acceleration.y])

        obs = build_state_vector(self.ego_vehicle, waypoints, self.waypoints_size, self.lane_width, speed, accel, self.max_dist_ahead)
        # ensure numpy float32 array
        return np.array(obs, dtype=np.float32)

    # ---------------------- reward ----------------------
    def compute_reward(self, waypoints=None):
        """Stateless potential-based reward using stored self.destination.

        Reward is higher when closer to the fixed destination. We also add small
        penalties for parking and collisions. No prev-distance is required.
        """
        reward = 0.0
        terminated = False

        if self.destination is None:
            logging.warning("compute_reward called before destination set")
            return -1.0, False

        if waypoints is None:
            waypoints = self.agent.get_waypoints()

        # ego position
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_pos = np.array([ego_loc.x, ego_loc.y], dtype=np.float32)

        # destination
        dest = self.destination
        dest_pos = np.array([dest.x, dest.y], dtype=np.float32)

        # distance to goal
        dist_to_goal = float(np.linalg.norm(dest_pos - ego_pos))
        max_dist = float(getattr(self, 'max_route_dist', 200.0))
        norm_dist = np.clip(dist_to_goal / max_dist, 0.0, 1.0)

        # potential-based reward (stateless): closer -> larger
        reward += (1.0 - norm_dist) * 2.0

        # small anti-parking penalty
        vel = self.ego_vehicle.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y])
        if speed < 0.2:
            reward -= 1.0

        # collisions & pedestrians
        wp = waypoints[0] if waypoints else None
        if wp:
            try:
                vehicle_state, vehicle, dist_v = self.agent.collision_and_car_avoid_manager(waypoint=wp)
                if vehicle_state:
                    reward -= 0.5
                    if dist_v < 5.0:
                        reward -= 50.0
                        terminated = True
            except Exception:
                logging.exception("vehicle collision check failed")

            try:
                walker_state, walker, dist_w = self.agent.pedestrian_avoid_manager(waypoint=wp)
                if walker_state:
                    reward -= 2.0
                    if dist_w < 5.0:
                        reward -= 100.0
                        terminated = True
            except Exception:
                logging.exception("pedestrian check failed")

        # tailgating
        try:
            if getattr(self.agent.behavior, 'tailgate_counter', 0) > 0:
                reward -= 1.0
        except Exception:
            pass

        # goal reached
        goal_thresh = float(getattr(self, 'goal_threshold', 3.0))
        if dist_to_goal <= goal_thresh or getattr(self.agent, 'done', lambda: False)():
            reward += 100.0
            terminated = True

        # clip extremes
        reward = float(np.clip(reward, -200.0, 200.0))
        return reward, terminated
