import logging
import random
import numpy as np
import Pyro4
import carla
import gymnasium as gym
import pygame

from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from env_utils import build_state_vector, get_vehicle_speed_accel
from simulation.python_3_8_20_scripts.camera_control import CameraManager, LiDARManager
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
from visualization.MinimalHUD import MinimalHUD

# Initialize Logger
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Constants
MAX_STEPS = 1500
WAIT_TICKS = 30
MAP_CHANGE_FREQ = 25

@Pyro4.expose
class CarlaEnv(gym.Env):
    """
    CARLA Gym Environment for DRL-based Autonomous Driving.
    This environment uses a CARLA client to provide a Gym-compatible interface for training DRL agents.
    The agent controls throttle and brake, while steering is managed by a Local Planner.

    Observation Space:
        - Future Waypoints (15 x 2): Local coordinates (x, y) of the next 15 waypoints.
        - Speed (1): Current speed of the vehicle (m/s).
        - Acceleration (1): Current acceleration of the vehicle (m/s²).
        - Steering (1): Current steering command applied by the agent [-1, 1].
        - Distance to vehicle Ahead (1): Distance to the vehicle object ahead (m).
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

        # Map configuration
        self.maps = ['Town01_Opt', 'Town02_Opt', 'Town03_Opt', 'Town04_Opt']
        """
        Training maps:
        Town01 = A small, simple town with a river and several bridges.
        Town02 = A small simple town with a mixture of residential and commercial buildings.
        Town03 = A larger, urban map with a roundabout and large junctions.
        Town04 = A small town embedded in the mountains with a special "figure of 8" infinite highway.

        Test map: 
        Town05 = Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
        """
        self.map_switch_freq = MAP_CHANGE_FREQ  # Switch every 50 steps
        self.total_episodes = 0

        # Setup vehicle and sensors
        self.ego_vehicle = None
        self.col_sensor = None
        self.camera_width = 800
        self.camera_height = 600
        self.camera_manager = None
        self.max_lidar_points = 120000
        self.lidar_manager = None

        ## Setup shared memory
        self.shared_memory_filepath = "/dev/shm/carla_shared/carla_shared_v6.dat"
        self.shared_memory = CarlaWrapper(filename=self.shared_memory_filepath,
                                          image_width=self.camera_width,
                                          image_height=self.camera_height,
                                          max_lidar_points=self.max_lidar_points)

        # Planners
        self.grp = None
        self.lp = None

        # Action: [Throttle/Brake] (Steering is handled by LocalPlanner)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.waypoints_size = 15
        self.lane_width = 3

        # Obs space size: Path (15*2) + Speed(1) + Accel(1) + Steer(1) + CTE(1) + DistAhead(1) = 35
        obs_size = self.waypoints_size * 2 + 5
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        #
        self.episode_step = 0
        self.distance_ahead = 50.0
        self.collision_history = []

        # Camera
        self.latest_rgb = None  # type: #Optional[np.ndarray]
        self.latest_frame_id = -1  # no frame received yet

        #
        self.traffic_actors = None
        self._init_world_settings()
        self._init_traffic_manager()
        self._load_map("Town05_Opt") # Start on first map

        logger.info("Carla environment initialized")

    def _init_world_settings(self):
        """Apply settings after loading a map"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _init_traffic_manager(self):
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(3)

    def _load_map(self, map_name: str):
        SPAWN_VEHICLES = 15
        logger.info(f"Loading map: {map_name}")
        self._cleanup()
        self.client.load_world(map_name)
        self.world = self.client.get_world()
        self._init_world_settings()
        self._spawn_ego_vehicle()
        self._spawn_traffic(SPAWN_VEHICLES)
        self.world.tick()


    def _spawn_ego_vehicle(self):

        self._cleanup_sensors()

        bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        spawn = random.choice(self.world.get_map().get_spawn_points())

        for _ in range(10):
            self.ego_vehicle = self.world.try_spawn_actor(bp, spawn)
            if self.ego_vehicle:
                break

        if self.ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle after 10 attempts")

        self._setup_sensors()

        self.grp = GlobalRoutePlanner(self.world.get_map(), 2.0)
        self.lp = LocalPlanner(self.ego_vehicle, {
            'target_speed': 50,
            'sampling_radius': 3.0,
            'lateral_control_dict': {'K_P': 1.0, 'K_D': 0.05, 'dt': 0.05}
        })

    def _reset_ego_pose(self):

        # If ego is gone, respawn instead of resetting
        if self.ego_vehicle is None or not self.ego_vehicle.is_alive:
            logger.warning("Ego vehicle missing, respawning.")
            self._cleanup_vehicle()
            self._spawn_ego_vehicle()
            return

        spawn = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle.set_transform(spawn)
        self.ego_vehicle.set_simulate_physics(True)

        self.ego_vehicle.apply_control(
            carla.VehicleControl(throttle=0, brake=1.0, hand_brake=True)
        )

        for _ in range(WAIT_TICKS):
            self.world.tick()

    def _setup_sensors(self):
        """ Sets up the vehicle sensors: Collision sensor and RGB camera. """

        # Collision
        bp_col = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(
            bp_col, carla.Transform(), attach_to=self.ego_vehicle)
        self.col_sensor.listen(self.collision_callback)

        # Camera
        self.camera_manager = CameraManager(
            client=self.client,
            world=self.world,
            parent_actor=self.ego_vehicle,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            shared_memory=self.shared_memory
        )

        # Lidar
        self.lidar_manager = LiDARManager(client=self.client,
                             world=self.world,
                             parent_actor=self.ego_vehicle,
                             shared_memory=self.shared_memory,
                             range_m=50.0,
                             channels=32,
                             points_per_second=100000,
                             rotation_frequency=20.0,
                             z_offset=1.73
                             )

        logging.info("Sensors setup complete.")

    def collision_callback(self, event):
        self.collision_history.append(event)

    def camera_callback(self, image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Drop alpha channel & convert BGRA -> RGB
        rgb = array[:, :, :3][:, :, ::-1].copy()
        self.latest_rgb = rgb
        self.latest_frame_id = image.frame

    def _spawn_traffic(self, n_vehicles):
        """
        Spawns vehicles in the simulation.
        - Filters vehicle blueprints to exclude bikes and motorcycles.
        - Spawns vehicles at random spawn points away from the ego vehicle.
        - Enables autopilot for spawned vehicles via Traffic Manager.
        - Uses batch commands for efficient spawning.

        :param n_vehicles: Number of cars to spawn.
        """

        logging.info(f"Spawning {n_vehicles} vehicles.")

        # 2. Get Blueprints
        bp_lib = self.world.get_blueprint_library()

        # Filter safe vehicles (no bikes/motorcycles/trucks usually better for training)
        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) > 3]

        # 3. Spawn Vehicles
        ego_loc = self.ego_vehicle.get_location()
        spawn_points = [
            sp for sp in self.world.get_map().get_spawn_points()
            if sp.location.distance(ego_loc) > 5.0
        ]
        n_vehicles = min(n_vehicles, len(spawn_points))
        random.shuffle(spawn_points)

        batch = []
        for i in range(n_vehicles):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            # Set Autopilot ON immediately
            bp.set_attribute('role_name', 'autopilot')

            # Prepare spawn command
            batch.append(carla.command.SpawnActor(bp, spawn_points[i])
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm.get_port())))

        # Execute Batch
        results = self.client.apply_batch_sync(batch, True)

        # Collect actors
        self.traffic_actors = [r.actor_id for r in results if not r.error]
        logger.info(f"Successfully spawned {len(self.traffic_actors)} vehicles.")


    def _get_gt_distance(self, range_limit=50.0):
        """
        Calculates the distance to the nearest vehicle in the same lane using CARLA ground truth data
        - Uses vector math to determine relative positions.
        - Considers only vehicles in front and within lane boundaries.
        - Optimized with a quick Euclidean distance check to skip far vehicles.
        - Returns the closest distance found, or the range limit if none found.

        :param range_limit: Maximum range to consider (meters)
        """
        if not self.ego_vehicle:
            return range_limit

        # 1. Get Ego Transform
        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
        ego_right = ego_tf.get_right_vector()

        # 2. Get all other vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')

        closest_dist = range_limit

        for target in vehicles:
            if target.id == self.ego_vehicle.id:
                continue  # Skip ourselves

            target_loc = target.get_transform().location

            # Optimization: Quick Euclidian check to skip far cars
            if ego_loc.distance(target_loc) > range_limit:
                continue

            # 3. Vector Math: Calculate relative position
            # Vector from Ego -> Target
            vec_to_target = carla.Vector3D(
                target_loc.x - ego_loc.x,
                target_loc.y - ego_loc.y,
                target_loc.z - ego_loc.z
            )

            # Project onto Forward Vector (How far ahead?)
            # Dot Product: A . B
            forward_dist = (vec_to_target.x * ego_fwd.x) + \
                           (vec_to_target.y * ego_fwd.y)

            # Project onto Right Vector (How far sideways?)
            # Dot Product with Right Vector gives lateral offset
            lateral_dist = (vec_to_target.x * ego_right.x) + \
                           (vec_to_target.y * ego_right.y)

            # 4. Check if it's in our "Corridor"
            # - Must be in front (forward_dist > 0)
            # - Must be closer than current closest (forward_dist < closest_dist)
            # - Must be in our lane (abs(lateral_dist) < half_lane_width)
            #   Assuming lane width ~3.5m, half is 1.75m. We use 1.5m to be strict.
            if 0 < forward_dist < closest_dist:
                if abs(lateral_dist) < 1.5:
                    closest_dist = forward_dist

        return closest_dist

    def get_waypoints(self):
        """ Returns the next waypoints from the Local Planner. """
        plan = self.lp.get_plan()
        waypoints = [wp[0] for wp in plan]
        wps = waypoints[:self.waypoints_size]

        return wps, waypoints

    def reset(self, seed=None, options=None) -> tuple:
        """
        Resets the environment for a new episode.
        - Switches map every N episodes.
        - Respawns or resets the ego vehicle.
        - Settles physics by applying brakes for a few ticks.
        - Generates a new random route using the Global Route Planner.
        - Clears collision history and resets episode step counter.
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

                self._load_map(self.maps[map_index])

            else:
                self._reset_ego_pose()

            # 2. Reset Actor State
            if not self.ego_vehicle or not self.ego_vehicle.is_alive:
                self._spawn_ego_vehicle()

            # 3. Settle Physics
            for _ in range(WAIT_TICKS):
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                self.world.tick()

            # 4. Route Generation
            # Smart destination: try to find a point far away
            dest_tf = random.choice(self.world.get_map().get_spawn_points())
            route = self.grp.trace_route(self.ego_vehicle.get_location(), dest_tf.location)
            self.lp.set_global_plan(route)

            self.collision_history.clear()
            self.distance_ahead = self._get_gt_distance()
            self.episode_step = 0

            waypoints, _ = self.get_waypoints()
            obs = self._get_obs(waypoints)

            self.safety_brake = 0

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

            # 2. STEERING
            lp_control = self.lp.run_step()
            steer = lp_control.steer

            # Emergency braking
            speed, _ = get_vehicle_speed_accel(self.ego_vehicle)

            max_decel = 5.0  # m/s² (comfortable emergency braking)
            stopping_dist = speed ** 2 / (2 * max_decel)

            if self.distance_ahead < stopping_dist + 2.0:
                throttle = 0.0
                brake = 1.0
                self.safety_brake = 1

            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            self.ego_vehicle.apply_control(control)

            # 3. TICK
            self.world.tick()
            self.episode_step += 1

            # 4. GET WAYPOINTS & DISTANCE
            waypoints, all_waypoints = self.get_waypoints()
            self.distance_ahead = self._get_gt_distance()

            # 5. VISUALIZATION
            if self.episode_step % 2 == 0 and all_waypoints:
                # Visualize the last waypoint to see destination reached
                self.world.debug.draw_point(
                    all_waypoints[-1].transform.location + carla.Location(z=2),
                    size=0.5, color=carla.Color(0, 0, 255), life_time=0.1
                )

            # 6. OBSERVATION & REWARD
            obs = self._get_obs(waypoints)
            reward, terminated = self._compute_reward(obs)
            truncated = self.episode_step >= MAX_STEPS

            self._render_hud(reward)
            self.info = {}
            return obs.tolist(), float(reward), bool(terminated), bool(truncated), self.info

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

    def _compute_reward(self, obs: np.array) -> tuple:
        """
        Computes the reward for the current state.
        - Penalizes collisions heavily.
        - Encourages going the speed limit unless in traffic.
        - Encourages staying in the lane.
        - Encourages smooth driving (low G-forces).
        - Penalize stopping unless in traffic.
        - Rewards reaching the goal.
        -> finally normalize reward
        :return: Tuple of (reward: float, terminated: bool)
        """

        # -------------------
        # 1. Collision
        # -------------------
        # if collision happened after initial wait period -> big mistake -> wrap it up
        if len(self.collision_history) > 0 and self.episode_step > WAIT_TICKS:
            return -200.0, True

        # --------------------------------------------------
        # 2. speed, speed limit, safe distance, distance ahead
        # --------------------------------------------------
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)
        speed_limit = self.ego_vehicle.get_speed_limit() / 3.6

        # safe distance = max(10m, 2 s * speed)
        safe_dist = max(10.0, speed * 2.0)
        dist_to_lead = self.distance_ahead

        car_ahead = dist_to_lead < safe_dist
        road_clear = dist_to_lead > 45.0

        # example
        # speed = 30 km/h = 8 m/s
        # safe_dist = 8*2 = 16 m
        # distance ahead = 15 m
        # is_blocked = True

        reward = 0.0
        terminated = False

        # --------------------------------------------------
        # 3. Target Speed & reward
        # --------------------------------------------------
        if car_ahead:
            # Linearly scale target speed based on distance to lead vehicle
            # Ratio keeps decreasing -> deceleration
            follow_ratio = np.clip(dist_to_lead / safe_dist, 0.0, 1.0)
            target_speed = follow_ratio * speed_limit

            # if very close, force full stop
            if dist_to_lead < 10:
                target_speed = 0.0

            # if too close, big penalty
            if dist_to_lead < 5.0:
                reward -= 5.0 * (4.0 - dist_to_lead)

        else:
            # else just go the speed limit
            target_speed = speed_limit

        # reward based on speed error
        speed_error = (speed - target_speed) / max(speed_limit, 1e-3)
        if speed <= target_speed:
            # reward underspeeding linearly
            r_speed = 1.0 - abs(speed_error)
        else:
            # penalize overspeeding harshly
            r_speed = -3.0 * speed_error ** 2
        reward += r_speed


        # --------------------------------------------------
        # 5. Progress reward
        # --------------------------------------------------
        # if no car ahead, reward progress towards goal
        if not car_ahead:
            reward += 0.5 * (speed / speed_limit)

        # --------------------------------------------------
        # 6. Penalties (Stop, Lane, G-Force)
        # --------------------------------------------------
        # Lazy Stop
        # if speed is less than half of the speed limit and road is clear, penalize
        if speed < 0.5 * speed_limit and road_clear:
            reward -= 2.0

        # Lane
        cte = obs[-2] * self.lane_width  # denormalize
        reward -= 0.4 * (cte ** 2)

        # Comfort
        accel = self.ego_vehicle.get_acceleration()
        tf = self.ego_vehicle.get_transform()

        # Project vectors
        fwd = tf.get_forward_vector()
        right = tf.get_right_vector()

        long_g = abs(accel.x * fwd.x + accel.y * fwd.y + accel.z * fwd.z) / 9.81
        lat_g = abs(accel.x * right.x + accel.y * right.y + accel.z * right.z) / 9.81

        reward -= 0.5 * lat_g
        reward -= 0.2 * max(0.0, long_g - 0.5)

        # --------------------------------------------------
        # 7. Clipping
        # --------------------------------------------------
        reward = np.clip(reward, -3.0, 3.0)

        # --------------------------------------------------
        # 8. Goal
        # --------------------------------------------------
        if self.lp.done():
            reward += 50.0
            terminated = True

        return reward, terminated

    def _cleanup(self):
        """Destroys current actors cleanly."""
        # SAFE CLEANUP
        self._cleanup_vehicle()
        self._cleanup_sensors()
        self._cleanup_traffic()

    def _cleanup_sensors(self):
        if self.col_sensor and self.col_sensor.is_alive:
            if self.col_sensor.is_listening: self.col_sensor.stop()
            self.col_sensor.destroy()
            self.col_sensor = None

        if self.camera_manager is not None:
            for s in self.camera_manager.sensors:
                if s is not None and s.is_alive:
                    if s.is_listening:
                        s.stop()
                    s.destroy()
            self.camera_manager = None

        if self.lidar_manager is not None:
            self.lidar_manager.destroy()
            self.lidar_manager = None

    def _cleanup_vehicle(self):
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def _cleanup_traffic(self):
        if self.traffic_actors is not None:
            batch = [carla.command.DestroyActor(x) for x in self.traffic_actors]
            self.client.apply_batch(batch, True)
            self.traffic_actors = None


    def close(self):
        """
        Cleans up actors, sensors and other resources.
        :return: None
        """
        self._cleanup()

    # Debugging purposes: Render HUD
    def _render_hud(self, reward: float):
        # 1. Get Real Control Values (What the car is actually doing)
        ctrl = self.ego_vehicle.get_control()
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)
        accel = self.ego_vehicle.get_acceleration()
        # 2. Get Car Location to place text
        loc = self.ego_vehicle.get_location()

        gforce = np.sqrt(accel.x ** 2 + accel.y ** 2) / 9.81

        # 3. Define the text lines (Bottom to Top)
        info_text = [
            f"Speed: {speed * 3.6:.1f} km/h",
            f"speed_limit: {self.ego_vehicle.get_speed_limit():.1f} km/h",
            f"Steer: {ctrl.steer:.2f}",
            f"Throttle: {ctrl.throttle:.2f} | Brake: {ctrl.brake:.2f}",
            f"GForce: {gforce:.2f} g | safety brake: {self.safety_brake:.2f}",
            f"distAhead: {self.distance_ahead:.2f} m",
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
    def get_latest_image(self):
        frame = self.shared_memory.read_latest_image()
        if frame is None:
            return None, None, None

        h, w, c = frame.shape
        img_bytes = frame.tobytes()

        # frame_id: if your shared memory wrapper stores it, return it; otherwise 0
        return img_bytes, (h, w, c), 0

    def get_latest_lidar_points(self):
        points = self.shared_memory.read_latest_lidar_points()
        if points is None:
            return None, None

        # points is (N, 4) float32
        return points.tobytes(), points.shape


    def draw_detections(self, detections, img_width=800, img_height=600):
        """
        Draw 3D debug boxes in the CARLA world based on 2D YOLO detections.

        detections: list of dicts:
            {
                "name": str,
                "conf": float,
                "bbox": [x1, y1, x2, y2]  # pixel coords in the camera image
            }

        img_width, img_height: resolution of the RGB camera image.
        """
        if self.ego_vehicle is None:
            return

        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()

        # Heuristic parameters
        max_lateral = 6.0  # max +/- meters from center
        min_dist = 8.0  # minimum distance in front of car
        max_extra = 20.0  # extra distance when object is high in image

        for det in detections:
            name = str(det.get("name", "obj"))
            conf = float(det.get("conf", 0.0))
            x1, y1, x2, y2 = det["bbox"]

            # Center of the bbox in pixel coords
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # Normalize horizontal position to [-1, 1]
            nx = (cx / img_width) * 2.0 - 1.0  # left=-1, center=0, right=+1
            # Normalize vertical position to [0, 1] (0=top, 1=bottom)
            ny = cy / img_height

            # Lateral offset: left/right in car frame
            offset_y = nx * max_lateral

            # Distance ahead: closer when near bottom of image
            dist_ahead = min_dist + (1.0 - ny) * max_extra

            # World position of the box center
            center = carla.Location(
                x=ego_loc.x + fwd.x * dist_ahead + right.x * offset_y,
                y=ego_loc.y + fwd.y * dist_ahead + right.y * offset_y,
                z=ego_loc.z + 1.5
            )

            # Scale box size with bbox height (pure heuristic)
            box_pix_height = max(y2 - y1, 1.0)
            box_height_m = 1.0 + 3.0 * (box_pix_height / img_height)  # between ~1m and ~4m

            extent = carla.Vector3D(0.8, box_height_m * 0.5, box_height_m * 0.5)
            bbox = carla.BoundingBox(center, extent)

            # Draw box
            self.world.debug.draw_box(
                bbox,
                ego_tf.rotation,  # aligned with ego
                thickness=0.1,
                color=carla.Color(0, 0, 255),  # green
                life_time=0.2
            )

            # Draw label above the box
            label = f"{name} {conf:.2f}"
            self.world.debug.draw_string(
                center + carla.Location(z=extent.z + 0.5),
                label,
                draw_shadow=False,
                color=carla.Color(0, 0, 255),
                life_time=0.2
            )

