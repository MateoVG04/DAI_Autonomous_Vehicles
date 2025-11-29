import logging
import random
import numpy as np
import Pyro4
import carla
import gymnasium as gym

from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner

from env_utils import build_state_vector, get_vehicle_speed_accel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

@Pyro4.expose
class CarlaEnv(gym.Env):
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        super().__init__()

        self.world = world
        self.ego_vehicle = vehicle

        # Action: [Throttle/Brake] (Steering is handled by LocalPlanner)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.waypoints_size = 15
        self.lane_width = 3.5

        # Obs: Path (15*2) + Speed(1) + Accel(1) + Steer(1) + CTE(1) + Dist(1) = 35
        obs_size = self.waypoints_size * 2 + 5
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # 1. SETUP PLANNERS
        # Global: Calculates the map path once
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=3.0)

        # Local: Follows the path dynamically
        # sampling_radius=2.0 ensures smooth curves. 3.0 is okay, but 2.0 is tighter.
        self.lp = LocalPlanner(self.ego_vehicle, opt_dict={
            'sampling_radius': 2.0,
            'lateral_control_dict': {'K_P': 1.0, 'K_D': 0.05, 'dt': 0.05}  # Soft steering
        })

        # Simulation settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # Sensors
        self.max_dist_ahead = 50.0
        self.collision_history = []
        self._setup_sensors()

        self.episode_step = 0

        logger.log(logging.INFO, "CarlaEnv initialized")

    def _setup_sensors(self):
        # Radar
        bp_radar = self.world.get_blueprint_library().find('sensor.other.radar')
        bp_radar.set_attribute('horizontal_fov', '30')
        bp_radar.set_attribute('range', '50')
        self.radar_sensor = self.world.spawn_actor(
            bp_radar, carla.Transform(carla.Location(x=2.5, z=1.0)), attach_to=self.ego_vehicle)
        self.radar_sensor.listen(self.radar_callback)

        # Collision (Replaces SteeringAgent check)
        bp_col = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(
            bp_col, carla.Transform(), attach_to=self.ego_vehicle)
        self.col_sensor.listen(self.collision_callback)

    def radar_callback(self, data):
        try:
            depths = [d.depth for d in data]
            self.max_dist_ahead = float(min(depths)) if depths else 50.0
        except:
            pass

    def collision_callback(self, event):
        self.collision_history.append(event)

    def reset(self, seed=None, options=None):
        try:
            # 1. Spawn & Dest
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

            # Simple dest logic: Pick a point far away or random
            dest_tf = random.choice(spawn_points)

            # 2. Apply Physics
            self.ego_vehicle.set_transform(spawn_point)
            self.ego_vehicle.set_simulate_physics(True)

            for _ in range(50):  # Let physics settle
                self.world.tick()
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            # 3. CALCULATE ROUTE & FEED LOCAL PLANNER
            route = self.grp.trace_route(spawn_point.location, dest_tf.location)
            self.lp.set_global_plan(route)

            # Reset controls
            self.max_dist_ahead = 50.0
            self.episode_step = 0
            self.collision_history = []

            # 4. GET OBSERVATION
            plan = self.lp.get_plan()

            waypoints = [wp[0] for wp in plan]
            wps = waypoints[:self.waypoints_size]

            obs = self.get_obs(wps)
            return obs.tolist(), {}
        except Exception as e:
            logger.exception("Reset Error")
            raise e

    def step(self, action):
        try:
            # 1. ACTION
            a = float(np.array(action).squeeze())
            throttle = float(np.clip(a, 0.0, 1.0)) if a > 0 else 0.0
            brake = float(np.clip(-a, 0.0, 1.0)) if a <= 0 else 0.0

            # 2. STEERING (From Local Planner)
            lp_control = self.lp.run_step()
            steer = lp_control.steer

            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            self.ego_vehicle.apply_control(control)

            # 3. TICK
            self.world.tick()
            self.episode_step += 1

            # 4. GET WAYPOINTS
            plan = self.lp.get_plan()
            waypoints = [wp[0] for wp in plan]
            wps = waypoints[:self.waypoints_size]

            # 5. VISUALIZATION
            if self.episode_step % 2 == 0 and waypoints:
                # Visualize the first valid one to ensure it matches
                self.world.debug.draw_point(
                    waypoints[-1].transform.location + carla.Location(z=2),
                    size=0.5, color=carla.Color(0, 0, 255), life_time=0.1
                )

            # 6. OBSERVATION & REWARD
            obs = self.get_obs(wps)
            reward, terminated = self.compute_reward(wps)
            truncated = self.episode_step > 1500

            self._render_hud(reward)

            info = {}
            return obs.tolist(), float(reward), bool(terminated), bool(truncated), info

        except Exception as e:
            logger.exception("Step Error")
            raise e

    def get_obs(self, waypoints):

        speed, accel = get_vehicle_speed_accel(self.ego_vehicle)
        steer = self.ego_vehicle.get_control().steer

        obs = build_state_vector(
            self.ego_vehicle, waypoints, self.waypoints_size, self.lane_width,
            speed, accel, steer, self.max_dist_ahead
        )

        return np.array(obs, dtype=np.float32)

    def compute_reward(self, waypoints):
        reward = 0.0
        terminated = False

        # Collision Check and avoid it at the start
        if len(self.collision_history) > 0 and self.episode_step > 20:
            reward -= 200.0
            terminated = True
            return reward, terminated

        # Speed Reward
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)

        if waypoints and len(waypoints) > 0:
            # Calculate difference from target
            speed_diff = abs(speed - 15)

            r_speed = 1.0 - (speed_diff / 15)
            reward += max(0.0, r_speed)

            # Centering
            dist = waypoints[0].transform.location.distance(self.ego_vehicle.get_location())
            reward -= dist * 0.1
        else:
            reward -= 3.0  # Off road/Lost signal

        # Stopping Penalty
        if speed < 0.5:
            reward -= 2.0  # Stronger penalty for stopping

        # Goal Reached
        if self.lp.done():
            reward += 500.0
            terminated = True

        return reward, terminated

    def close(self):
        if self.radar_sensor: self.radar_sensor.destroy()
        if self.col_sensor: self.col_sensor.destroy()

    def _render_hud(self, reward):
        # 1. Get Real Control Values (What the car is actually doing)
        ctrl = self.ego_vehicle.get_control()
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)
        # 2. Get Car Location to place text
        loc = self.ego_vehicle.get_location()

        # 3. Define the text lines (Bottom to Top)
        info_text = [
            f"Speed: {speed:.1f} m/s",
            f"Steer: {ctrl.steer:.2f}",
            f"Throttle: {ctrl.throttle:.2f} | Brake: {ctrl.brake:.2f}",
            f"Reward: {reward:.2f}",
            f"Episode: {self.episode_step}"
        ]

        # 4. Draw the strings in the world
        # We stack them vertically by increasing Z
        for i, line in enumerate(info_text):
            self.world.debug.draw_string(
                # Position: 2 meters above car, slightly offset per line
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 2.0 + (i * 0.5)),
                line,
                draw_shadow=True,
                color=carla.Color(255, 255, 255),  # White text
                life_time=0.05  # Update every frame (assuming 20fps)
            )