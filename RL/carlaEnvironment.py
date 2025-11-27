import logging
import random
import numpy as np
import Pyro4
import carla
import gymnasium as gym

from env_utils import build_state_vector, get_vehicle_speed_accel
from steering_agent import SteeringAgent

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
        self.waypoints_size = 15
        self.lane_width = 3.5
        obs_size = self.waypoints_size * 2 + 5  # x,y per waypoint + speed + accel + steer + cte + dist
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

        logging.info("CarlaEnv initialized")


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
        self.ego_vehicle.set_simulate_physics(True)
        # ensure physics updated
        try:
            self.world.tick()
        except Exception:
            logging.exception("world.tick() failed in reset")

        # inform agent (global/local planner)
        try:
            self.agent.set_destination(destination.location)
        except Exception:
            logging.exception("agent.set_destination failed")

        # reset vehicle control and sensors
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        self.max_dist_ahead = 50.0
        self.episode_step = 0

        # observations and reward
        all_waypoints = self.agent.get_waypoints()  # fresh local window

        # get current waypoints up to waypoints_size
        current_waypoints = all_waypoints[:self.waypoints_size]
        # initial observation
        obs = self.get_obs(current_waypoints)
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
            all_waypoints = self.agent.get_waypoints()  # fresh local window

            # get current waypoints up to waypoints_size
            current_waypoints = all_waypoints[:self.waypoints_size]
            if current_waypoints:
                # 1. Print to console ONCE per second to prove we have data
                if self.episode_step % 20 == 0:
                    print(f"DEBUG: Drawing {len(current_waypoints)} waypoints...")

                for i, wp in enumerate(current_waypoints):

                    loc = wp.transform.location

                    # 2. DRAW GIANT PILLARS
                    # We draw a line from the road surface (z) up to the air (z+2)
                    # This makes a "fence" that is impossible to miss.
                    self.world.debug.draw_line(
                        carla.Location(loc.x, loc.y, loc.z),  # Start at feet
                        carla.Location(loc.x, loc.y, loc.z + 3.0),  # End in air
                        thickness=0.2,
                        color=carla.Color(255, 0, 0),  # Bright RED
                        life_time=0.1  # Update every frame
                    )

                    # 3. Add a ball on top
                    self.world.debug.draw_point(
                        carla.Location(loc.x, loc.y, loc.z + 3.0),
                        size=0.5,
                        color=carla.Color(0, 255, 0),  # Green top
                        life_time=0.1
                    )
            else:
                # If this prints, your agent is lost and has no target
                if self.episode_step % 20 == 0:
                    print("!!! AGENT HAS NO WAYPOINTS !!!")

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
    def get_obs(self, waypoints):

        speed, accel = get_vehicle_speed_accel(self.ego_vehicle)

        steering = self.agent.vehicle.get_control().steer
        obs = build_state_vector(self.ego_vehicle, waypoints, self.waypoints_size, self.lane_width,
                                 speed, accel, steering, self.max_dist_ahead)

        # ensure numpy float32 array
        return np.array(obs, dtype=np.float32)

    # ---------------------- reward ----------------------
    def compute_reward(self, waypoints):
        """Stateless potential-based reward using stored self.destination.

        Reward is higher when closer to the fixed destination. We also add small
        penalties for parking and collisions. No prev-distance is required.
        """
        reward = 0.0
        terminated = False

        # 1. Get State
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()

        # 2. Progress Reward (Speed along the lane)
        # Instead of distance to goal, we look at speed projected onto the road direction
        vehicle_velocity = self.ego_vehicle.get_velocity()
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)

        if waypoints and len(waypoints) > 0:
            # Vector pointing to the next waypoint
            wp_fwd = waypoints[0].transform.rotation.get_forward_vector()

            # Dot product: How much of our speed is actually heading toward the waypoint?
            # If we are going 20m/s but 90 degrees wrong, this rewards 0.
            speed_reward = (vehicle_velocity.x * wp_fwd.x) + (vehicle_velocity.y * wp_fwd.y)
            reward += speed_reward * 0.1  # Scale factor

            # 3. Centering Penalty (Cross Track Error)
            # Calculate distance from ego to the waypoint line
            # Simple approximation: distance to the nearest waypoint
            dist_to_center = ego_loc.distance(waypoints[0].transform.location)
            reward -= dist_to_center * 0.1
        else:
            # Fallback if no waypoints found
            reward -= 1.0

        # small anti-parking penalty
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
        if self.agent.done():
            reward += 100.0
            terminated = True

        return reward, terminated
