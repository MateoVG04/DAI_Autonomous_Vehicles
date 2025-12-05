import logging
import random
import numpy as np
import Pyro4
import carla
import gymnasium as gym

from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from env_utils import build_state_vector, get_vehicle_speed_accel

# Initialize Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Constants
EPISODES = 1500
WAIT_TICKS = 15


@Pyro4.expose
class CarlaEnv(gym.Env):
    """
    CARLA Gym Environment for DRL-based Autonomous Driving.
    This environment uses a CARLA vehicle and world instance to provide a Gym-compatible interface
    for training DRL agents. The agent controls throttle and brake, while steering is managed by a Local Planner.
    Observation Space:
        - Future Waypoints (15 x 2): Local coordinates of the next 15 waypoints.
        - Speed (1): Current speed of the vehicle (m/s).
        - Acceleration (1): Current acceleration of the vehicle (m/s²).
        - Steering (1): Current steering command applied by the agent [-1, 1].
        - Distance to Object Ahead (1): Distance to the nearest object ahead (m).
    Action Space:
        - Throttle/Brake (1): Continuous value in [-1, 1], where positive values indicate throttle
          and negative values indicate braking.
    Methods:
        - reset(): Resets the environment and returns the initial observation.
        - step(action): Applies the action, advances the simulation, and returns the new observation,
          reward, termination status, truncation status, and info.
        - close(): Cleans up sensors and other resources.
        - _get_obs(waypoints): Constructs the observation vector.
        - _compute_reward(waypoints): Computes the reward based on the current state.
    """
    def __init__(self, client: carla.Client):
        """
        Initializes the CARLA Gym Environment.
        :param: client: carla.Client, An active CARLA client connected to the simulator.
        """
        super().__init__()

        # Carla world and vehicle
        self.client = client
        self.world = self.client.get_world()

        # map configuration
        self.maps = ['Town01', 'Town02', 'Town03']
        self.map_switch_freq = 20  # Switch every 20 episodes
        self.total_episodes = 0

        # Setup vehicle and sensors
        self.ego_vehicle = None
        self.radar_sensor = None
        self.col_sensor = None

        # Planners
        self.grp = None
        self.lp = None

        # Action: [Throttle/Brake] (Steering is handled by LocalPlanner)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.waypoints_size = 15
        self.lane_width = 3.5

        # Obs space size: Path (15*2) + Speed(1) + Accel(1) + Steer(1) + CTE(1) + Dist(1) = 35
        obs_size = self.waypoints_size * 2 + 5
        self.obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.distance_ahead = 50.0
        self.collision_history = []
        self.episode_step = 0

        self._init_world_settings()
        self._setup_vehicle_and_sensors()

        logger.log(logging.INFO, "CarlaEnv initialized")

    def _init_world_settings(self):
        """Re-apply settings after loading a map"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _setup_vehicle_and_sensors(self):
        """Spawns ego vehicle and attaches sensors (Run after map load)"""
        self.cleanup()

        # 2. Spawn Vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if not self.ego_vehicle:
            raise RuntimeError("Failed to spawn vehicle")

        # 3. Setup Sensors (Radar/Collision)
        self._setup_sensors()

        # 4. Setup Planners
        # Global route planner: Calculates the map path once
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2.0)

        # Local: Follows the path dynamically
        self.lp = LocalPlanner(self.ego_vehicle, opt_dict={
            'target_speed': 54, # km/h
            'sampling_radius': 2.0,
            'lateral_control_dict': {'K_P': 1.0, 'K_D': 0.05, 'dt': 0.05}  # Soft steering
        })

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
            self.distance_ahead = float(min(depths)) if depths else 50.0
        except: pass

    def collision_callback(self, event):
        self.collision_history.append(event)

    def reset(self, seed=None, options=None) -> tuple:
        """
        Resets the environment for a new episode.
        - Increments episode count and switches map if needed.
        - Spawns the vehicle at a random spawn point and sets destination.
        - Calculates the route using the Global Route Planner and resets the Local Planner with the new route.
        - Returns the initial observation.

        :param seed: Random seed for reproducibility (not used)
        :param options: Additional options (not used)
        :return: Initial observation as a numpy array and an empty info dict

        :raises
            Exception: Any unexpected runtime error is logged and re-raised for debugging.
        """
        try:
            self.total_episodes += 1

            # 1. Map Switching
            if self.total_episodes > 1 and self.total_episodes % self.map_switch_freq == 0:
                map_index = (self.total_episodes // self.map_switch_freq) % len(self.maps)
                new_map_name = self.maps[map_index]
                logger.info(f"Switching Map to {new_map_name}...")

                self.client.load_world(new_map_name)
                self.world = self.client.get_world()
                self._init_world_settings()
                self._setup_vehicle_and_sensors()

            # 2. Reset Actor State
            if not self.ego_vehicle or not self.ego_vehicle.is_alive:
                self._setup_vehicle_and_sensors()

            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

            self.ego_vehicle.set_transform(spawn_point)
            self.ego_vehicle.set_simulate_physics(True)


            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            # 3. Settle Physics
            for _ in range(WAIT_TICKS):
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                self.world.tick()


            # 4. Route Generation
            # Smart destination: try to find a point far away
            dest_tf = random.choice(spawn_points)
            route = self.grp.trace_route(spawn_point.location, dest_tf.location)
            self.lp.set_global_plan(route)

            self.distance_ahead = 50.0
            self.episode_step = 0
            self.collision_history = []

            # 5. Get Observation (Using correct slicing)
            plan = self.lp.get_plan()
            waypoints = [wp[0] for wp in plan]
            wps = waypoints[:self.waypoints_size]
            obs = self._get_obs(wps)

            return obs.tolist(), {}
        except Exception as e:
            logger.exception("Step Error")
            raise e

    def step(self, action: list) -> tuple:
        """
        Execute one simulation step in the CARLA environment.
        - Gets action (throttle/brake) from the agent and steering from the Local Planner.
        - Applies the given action (throttle/brake) to the vehicle.
        - Advances the simulation by one tick.
        - Retrieves the next waypoints from the Local Planner.
        - Computes the observation, reward, and termination status.

        :param action: List containing a single float value in [-1, 1] for throttle/brake.
        :return: Tuple containing:
            - observation (np.ndarray): The next observation after the action.
            - reward (float): The reward obtained from this step.
            - terminated (bool): Whether the episode has terminated.
            - truncated (bool): Whether the episode has been truncated.
            - info (dict): Additional information.

        :raises
            Exception: Any unexpected runtime error is logged and re-raised for debugging.
        """
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
                # Visualize the last waypoint to see destination reached
                self.world.debug.draw_point(
                    waypoints[-1].transform.location + carla.Location(z=2),
                    size=0.5, color=carla.Color(0, 0, 255), life_time=0.1
                )

            # 6. OBSERVATION & REWARD
            obs = self._get_obs(wps)
            reward, terminated = self._compute_reward(wps)
            truncated = self.episode_step > EPISODES

            self._render_hud(reward)

            return obs.tolist(), float(reward), bool(terminated), bool(truncated), {}

        except Exception as e:
            logger.exception("Step Error")
            raise e

    def _get_obs(self, waypoints: list) -> np.ndarray:
        """
        Gets the current observation for the DRL agent.
        :param waypoints: The next waypoints from the Local Planner
        :return: Observation as a numpy array
        """

        speed, accel = get_vehicle_speed_accel(self.ego_vehicle)
        steer = self.ego_vehicle.get_control().steer

        obs = build_state_vector(
            self.ego_vehicle, waypoints, self.waypoints_size, self.lane_width,
            speed, accel, steer, self.distance_ahead
        )

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, waypoints: list) -> tuple:
        """
        Computes the reward for the current state.
        - Penalizes collisions heavily.
        - Encourages maintaining a target speed of 15 m/s and centering vehicle in lane.
        - Penalizes stopping and going off-road.
        - Rewards reaching the goal greatly.
        :param waypoints: The next waypoints from the Local Planner
        :return: Tuple of (reward: float, terminated: bool)
        """
        reward = 0.0
        terminated = False

        # Collision Check and avoid it at the start
        if len(self.collision_history) > 0 and self.episode_step > WAIT_TICKS:
            return -200.0, True # Heavy penalty for collision

        # Speed Reward
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)

        target_speed = 15.0  # m/s

        if waypoints and len(waypoints) > 0:

            # Calculate difference from target
            speed_diff = abs(speed - target_speed)
            r_speed = 1.0 - (speed_diff / target_speed)
            reward += max(0.0, r_speed)

            # Bell curve
            # r_speed = np.exp(-0.1 * speed_diff ** 2)  # Bell curve
            # reward += r_speed

            # Centering
            dist = waypoints[0].transform.location.distance(self.ego_vehicle.get_location())
            reward -= dist * 0.1
        else:
            reward -= 3.0  # Off-road/Lost signal

        # Stopping Penalty
        if speed < 0.5:
            reward -= 2.0  # Stronger penalty for stopping

        # Goal Reached
        if self.lp.done():
            reward += 500.0
            terminated = True

        return reward, terminated

    def cleanup(self):
        """Destroys current actors"""
        if self.radar_sensor and self.radar_sensor.is_alive:
            self.radar_sensor.destroy()
        if self.col_sensor and self.col_sensor.is_alive:
            self.col_sensor.destroy()
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()

    def close(self):
        """
        Cleans up actors, sensors and other resources.
        :return: None
        """
        self.cleanup()

    # Debugging purposes: Render HUD
    def _render_hud(self, reward: float):
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