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

        # Map configuration
        self.maps = ['Town01', 'Town02', 'Town03']
        self.map_switch_freq = 20  # Switch every 20 episodes
        self.total_episodes = 0

        # Setup vehicle and sensors
        self.ego_vehicle = None
        self.radar_sensor = None
        self.col_sensor = None
        self.camera_sensor = None

        # Planners
        self.grp = None
        self.lp = None

        # Action: [Throttle/Brake] (Steering is handled by LocalPlanner)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.waypoints_size = 15
        self.lane_width = 3.5

        # Obs space size: Path (15*2) + Speed(1) + Accel(1) + Steer(1) + CTE(1) + DistAhead(1) = 35
        obs_size = self.waypoints_size * 2 + 5
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        self.distance_ahead = 50.0
        self.latest_rgb = None  # type: #Optional[np.ndarray]
        self.latest_frame_id = -1  # no frame received yet
        self.collision_history = []
        self.episode_step = 0

        self.traffic_actors = []
        self._init_world_settings()
        self._setup_vehicle_and_sensors()

        logger.info("CarlaEnv initialized")

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

        # 4. Spawn Traffic
        self._spawn_traffic(10, 0)  # Spawn 10 vehicles, 0 walkers

        # 5. Setup Planners
        # Global route planner: Calculates the map path once
        self.grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=2.0)

        # Local: Follows the path dynamically
        self.lp = LocalPlanner(self.ego_vehicle, opt_dict={
            'target_speed': 54, # km/h
            'sampling_radius': 2.0,
            'lateral_control_dict': {'K_P': 1.0, 'K_D': 0.05, 'dt': 0.05}  # Soft steering
        })

    def _spawn_traffic(self, n_vehicles=10, n_walkers=0):
        """
        Spawns vehicles and walkers in the simulation.
        :param n_vehicles: Number of cars to spawn.
        :param n_walkers: Number of pedestrians to spawn.
        """
        logging.info(f"Spawning {n_vehicles} vehicles and {n_walkers} walkers...")

        # 1. Get Traffic Manager
        tm = self.client.get_trafficmanager(8000)  # Default port
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_hybrid_physics_mode(True)  # Optimization
        tm.set_synchronous_mode(True)

        # 2. Get Blueprints
        bp_lib = self.world.get_blueprint_library()

        # Filter safe vehicles (no bikes/motorcycles/trucks usually better for training)
        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) > 3]

        # 3. Spawn Vehicles
        spawn_points = self.world.get_map().get_spawn_points()
        for sp in spawn_points:
            if sp == self.ego_vehicle.get_transform():
                spawn_points.remove(sp)
                break
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
                         .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

        # Execute Batch
        results = self.client.apply_batch_sync(batch, True)

        # Check for failures (optional logging)
        self.traffic_actors = [r.actor_id for r in results if not r.error]
        logger.info(f"Successfully spawned {len(self.traffic_actors)} vehicles.")

        # 4. Spawn Walkers (Optional)
        if n_walkers > 0:
            # (Walker spawning is more complex: needs Controller + Walker actor)
            # Keeping it simple: Just spawn points navigation for now
            pass

    def _setup_sensors(self):

        # Collision
        bp_col = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(
            bp_col, carla.Transform(), attach_to=self.ego_vehicle)
        self.col_sensor.listen(self.collision_callback)

        bp_rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp_rgb_cam.set_attribute('image_size_x', '800')
        bp_rgb_cam.set_attribute('image_size_y', '600')
        bp_rgb_cam.set_attribute('fov', '90')
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=1.6),  # front & slightly above
            carla.Rotation(pitch=0.0)
        )
        self.camera_sensor = self.world.spawn_actor(
            bp_rgb_cam, cam_transform, attach_to=self.ego_vehicle
        )
        self.camera_sensor.listen(self.camera_callback)

    def radar_callback(self, data):
        try:
            depths = [d.depth for d in data]
            self.distance_ahead = float(min(depths)) if depths else 50.0
        except: pass

    def collision_callback(self, event):
        self.collision_history.append(event)

    def camera_callback(self, image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Drop alpha channel & convert BGRA -> RGB
        rgb = array[:, :, :3][:, :, ::-1].copy()
        self.latest_rgb = rgb
        self.latest_frame_id = image.frame

    def _get_gt_distance(self, range_limit=50.0):
        """
        Calculates the distance to the nearest vehicle in the same lane using CARLA ground truth data
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
                if abs(lateral_dist) < 1.75:
                    closest_dist = forward_dist

        return closest_dist

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

                self.cleanup()
                self.client.load_world(new_map_name)
                self.world = self.client.get_world()
                self._init_world_settings()
                self._setup_vehicle_and_sensors() # Respawn vehicle and sensors and traffic

            # 2. Reset Actor State
            if not self.ego_vehicle or not self.ego_vehicle.is_alive:
                self._setup_vehicle_and_sensors()

            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)

            self.ego_vehicle.set_transform(spawn_point)
            self.ego_vehicle.set_simulate_physics(True)

            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
            # 3. Settle Physics
            for _ in range(WAIT_TICKS):
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                self.world.tick()


            # 4. Route Generation
            # Smart destination: try to find a point far away
            dest_tf = random.choice(spawn_points)
            route = self.grp.trace_route(spawn_point.location, dest_tf.location)
            self.lp.set_global_plan(route)

            self.distance_ahead = self._get_gt_distance()
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

            # 2. STEERING
            lp_control = self.lp.run_step()
            steer = lp_control.steer

            control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
            self.ego_vehicle.apply_control(control)

            # 3. TICK
            self.world.tick()
            self.episode_step += 1

            # 4. GET WAYPOINTS & DISTANCE
            plan = self.lp.get_plan()
            waypoints = [wp[0] for wp in plan]
            wps = waypoints[:self.waypoints_size]
            self.distance_ahead = self._get_gt_distance()

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
        - Encourages going the speed limit and staying in lane.
        - Encourages safe following distance.
        - Encourages smooth driving (low G-forces).
        - Penalizes stopping and going off-road.
        - Rewards reaching the goal greatly.
        :param waypoints: The next waypoints from the Local Planner
        :return: Tuple of (reward: float, terminated: bool)
        """

        reward = 0.0
        terminated = False

        # 1. CRITICAL: Collision
        if len(self.collision_history) > 0 and self.episode_step > WAIT_TICKS:
            return -200.0, True

        # 2. Get Data
        speed, _ = get_vehicle_speed_accel(self.ego_vehicle)  # m/s

        # Dynamic Speed Limit
        speed_limit_kmh = self.ego_vehicle.get_speed_limit()
        target_speed = max(5.0, speed_limit_kmh / 3.6)

        # 3. ACC Logic
        # 2-second rule + buffer
        safe_dist = max(5.0, speed * 2.0)
        dist_to_lead = self.distance_ahead

        if waypoints:
            # --- FIX 1: Unified Reward Logic ---
            # Instead of separate branches, we just change the TARGET speed.

            if dist_to_lead < safe_dist:
                # TRAFFIC MODE:
                # If we are too close, the target speed is NOT the limit.
                # The target speed is ZERO (or matching the lead car).
                # This allows the "Speed Reward" below to reward us for slowing down.
                effective_target_speed = 0.0

                # Optional: Extra penalty for being dangerously close (Tailgating)
                if dist_to_lead < safe_dist * 0.5:
                    reward -= 1.0
            else:
                # FREE FLOW MODE:
                effective_target_speed = target_speed

            # --- Speed Reward (Applies to both modes) ---
            # If Traffic Mode: We get points for slowing to 0.
            # If Free Flow: We get points for hitting speed limit.
            speed_diff = abs(speed - effective_target_speed)
            r_speed = np.exp(-0.1 * speed_diff ** 2)
            reward += r_speed

            # Lane Centering
            dist_center = waypoints[0].transform.location.distance(self.ego_vehicle.get_location())
            reward -= dist_center * 0.1
        else:
            reward -= 2.0

        # 4. G-Force (FIX 2: Coordinate Transformation)
        # We must project the world acceleration onto the car's local vectors
        accel_vec = self.ego_vehicle.get_acceleration()
        ego_tf = self.ego_vehicle.get_transform()
        fwd_vec = ego_tf.get_forward_vector()
        right_vec = ego_tf.get_right_vector()

        long_accel = accel_vec.dot(fwd_vec)
        lat_accel = accel_vec.dot(right_vec)

        long_g = abs(long_accel) / 9.81
        lat_g = abs(lat_accel) / 9.81

        # Swerving (Lateral) is bad
        reward -= (lat_g ** 2) * 5.0

        # Braking (Longitudinal) is okay if necessary, but punish jerky driving
        if long_g > 0.5:
            reward -= (long_g ** 2) * 1.0

        # 5. Stopping Penalty (FIX 3: Context Aware)
        # Only punish stopping if the road is CLEAR.
        # If there is a car ahead (dist < 20), stopping is allowed (and rewarded by speed logic).
        if speed < 0.1 and dist_to_lead > 20.0:
            reward -= 5.0

        # Goal
        if self.lp.done():
            reward += 500.0
            terminated = True

        return reward, terminated

    def cleanup(self):
        """Destroys current actors cleanly."""
        # 1. CLEANUP SENSORS
        if self.col_sensor and self.col_sensor.is_alive:
            if self.col_sensor.is_listening:  # <--- CRITICAL CHECK
                self.col_sensor.stop()
            self.col_sensor.destroy()

        # 2. CLEANUP VEHICLE
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()

        # 3. CLEANUP TRAFFIC
        if not self.traffic_actors:
            return

        logger.info(f"Destroying {len(self.traffic_actors)} traffic vehicles...")

        # Use batch command for instant destruction
        batch = [carla.command.DestroyActor(x) for x in self.traffic_actors]
        self.client.apply_batch(batch)

        # Clear the list
        self.traffic_actors = []

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
        accel = self.ego_vehicle.get_acceleration()
        # 2. Get Car Location to place text
        loc = self.ego_vehicle.get_location()

        gforce = np.sqrt(accel.x ** 2 + accel.y ** 2) / 9.81

        # 3. Define the text lines (Bottom to Top)
        info_text = [
            f"Speed: {speed:.1f} m/s",
            f"Steer: {ctrl.steer:.2f}",
            f"Throttle: {ctrl.throttle:.2f} | Brake: {ctrl.brake:.2f}",
            f"GForce: {gforce:.2f} g",
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
        if self.latest_rgb is None:
            # nothing received yet
            return None, None, None

            # latest_rgb is a numpy array (H, W, 3), dtype uint8
        h, w, c = self.latest_rgb.shape
        img_bytes = self.latest_rgb.tobytes()  # flat uint8 buffer

        # Return only builtin types: bytes + tuple + int
        return img_bytes, (h, w, c), int(self.latest_frame_id)

    def draw_detections(self, detections):
        """
        Draws simple 3D boxes + labels in the CARLA world for each detection.
        `detections` is a list of dicts with keys: 'name' and 'conf'.
        """
        if self.ego_vehicle is None:
            return

        ego_loc = self.ego_vehicle.get_location()

        for i, det in enumerate(detections):
            label = f"{det['name']} {det['conf']:.2f}"

            # Place each box in front of the ego vehicle, but offset sideways
            # so they don't overlap (purely for visualization; not real 3D positions)
            offset_x = 8.0  # 8 m in front of the vehicle
            offset_y = (i - len(detections) / 2) * 2.0  # spread them left/right
            center = carla.Location(
                x=ego_loc.x + offset_x,
                y=ego_loc.y + offset_y,
                z=ego_loc.z + 1.5
            )

            # A small box representing the detection
            extent = carla.Vector3D(1.0, 1.0, 1.0)
            bbox = carla.BoundingBox(center, extent)

            # Draw green box
            self.world.debug.draw_box(
                bbox,
                carla.Rotation(),  # axis-aligned
                thickness=0.1,
                color=carla.Color(0, 255, 0),
                life_time=0.05  # refresh every tick
            )

            # Draw label above the box
            self.world.debug.draw_string(
                center + carla.Location(z=1.2),
                label,
                draw_shadow=True,
                color=carla.Color(255, 255, 255),
                life_time=0.05
            )