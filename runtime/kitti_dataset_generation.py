import logging
import queue
import random

import carla
import pygame
from charset_normalizer.md import getLogger

from agents.navigation.basic_agent import BasicAgent

print("CARLA loaded from:", carla.__file__)

import os
import numpy as np
import carla

# ==============================================================================
# -- DATASET --------------------------------------------------------------
# ==============================================================================
class KittiGenerator:
    def __init__(self, base_path, width, height, fov):
        self.base_path = base_path
        self.width = width
        self.height = height
        self.fov = fov
        self.idx = 0
        self.setup_directories()
        self.intrinsic = self._build_intrinsic_matrix()

    def setup_directories(self):
        """Creates the KITTI folder structure."""
        self.folders = {
            "calib": os.path.join(self.base_path, "training/calib"),
            "image": os.path.join(self.base_path, "training/image_2"),
            "label": os.path.join(self.base_path, "training/label_2"),
            "velodyne": os.path.join(self.base_path, "training/velodyne")
        }
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)

    def _build_intrinsic_matrix(self):
        """Builds camera intrinsic matrix (K)."""
        f = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        cx = self.width / 2.0
        cy = self.height / 2.0
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    def save_frame(self, image, lidar_data, vehicle, all_actors):
        """Main method to save all data for the current tick."""
        filename = f"{self.idx:06d}"

        # 1. Save Image
        image.save_to_disk(os.path.join(self.folders["image"], f"{filename}.png"))

        # 2. Save LiDAR (Bin format)
        # CARLA gives xyz, KITTI expects xyzi (intensity). We pad with 1.0 for intensity.
        lidar_data = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        n_points = int(lidar_data.shape[0] / 4)
        points = lidar_data.reshape((n_points, 4))
        # Swap columns to match KITTI if necessary (CARLA is usually fine, but KITTI is Z-up? No, KITTI is Z-forward)
        # Standard CARLA LiDAR is raw; usually needs standardizing. Saving raw here:
        points.tofile(os.path.join(self.folders["velodyne"], f"{filename}.bin"))

        # 3. Save Calibration
        self._save_calibration(filename)

        # 4. Save Labels (Bounding Boxes)
        self._save_labels(filename, vehicle, all_actors)

        print(f"Saved frame {filename}")
        self.idx += 1

    def _save_calibration(self, filename):
        """Saves a dummy calibration file (P2 is the critical one)."""
        # In a real setup, you calculate offsets between Lidar/Camera.
        # Here we assume sensors are at (0,0,0) relative to each other for simplicity or handle it via transforms.
        k_flat = self.intrinsic.reshape(1, 9)[0].tolist()
        # Extend K to 3x4 projection matrix (P2)
        p2 = [k_flat[0], k_flat[1], k_flat[2], 0,
              k_flat[3], k_flat[4], k_flat[5], 0,
              k_flat[6], k_flat[7], k_flat[8], 0]

        with open(os.path.join(self.folders["calib"], f"{filename}.txt"), 'w') as f:
            f.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("P1: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write(f"P2: {' '.join(map(str, p2))}\n")  # The one strictly used for Image_2
            f.write("P3: 0 0 0 0 0 0 0 0 0 0 0 0\n")
            f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")
            f.write("Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n")  # Standard CARLA->KITTI rotation
            f.write("Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0")

    def _save_labels(self, filename, ego_vehicle, actors):
        """Calculates 3D bbox in camera coordinates."""
        world_to_camera = np.array(ego_vehicle.get_transform().get_inverse_matrix())

        # Standard matrix to convert UE4 (x-fwd, y-right, z-up) to Camera (x-right, y-down, z-fwd)
        ue_to_cam = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        labels = []
        for npc in actors:
            if npc.id == ego_vehicle.id or "vehicle" not in npc.type_id:
                continue

            # Calculate relative transform
            npc_trans = npc.get_transform()
            loc_rel = self._apply_transform(npc_trans.location, world_to_camera)

            loc_rel[2] -= 1.7 # APPLY Z-OFFSET HERE"

            # Simple visibility check (is it in front of the camera?)
            if loc_rel[0] > 0:
                # Convert to KITTI coords
                loc_kitti = np.dot(ue_to_cam, np.append(loc_rel, 1))[:3]

                # Check if it projects onto image plane
                u, v, z = np.dot(self.intrinsic, loc_kitti)
                u, v = u / z, v / z

                if 0 <= u < self.width and 0 <= v < self.height:
                    # It's visible! Generate label line
                    # Format: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
                    # Filling 0 for truncated/occluded/alpha/bbox (bbox requires projecting 8 points) for brevity
                    labels.append(
                        f"Car 0 0 0 0 0 0 0 1.5 2.0 4.5 {loc_kitti[0]:.2f} {loc_kitti[1]:.2f} {loc_kitti[2]:.2f} 0")

        with open(os.path.join(self.folders["label"], f"{filename}.txt"), 'w') as f:
            f.write("\n".join(labels))

    def _apply_transform(self, location, transform_matrix):
        """Applies a 4x4 transform matrix to a 3D location."""
        point = np.array([location.x, location.y, location.z, 1])
        point_transformed = np.dot(transform_matrix, point)
        return point_transformed[:3]

# ==============================================================================
# -- HUD --------------------------------------------------------------
# ==============================================================================
class MinimalHUD:
    def __init__(self, width: int, height: int):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.clock = pygame.time.Clock()

        # We split the screen: Left = Camera, Right = LiDAR
        self.half_w = width // 2
        self.height = height

        # LiDAR Surface (Top-down view)
        self.lidar_surface = pygame.Surface((self.half_w, self.height))
        self.lidar_surface.set_alpha(255)  # Opaque

    def tick(self):
        self.clock.tick()

    def render(self, display, current_image, current_lidar, camera_sensor, vehicle, all_actors, camera_k):
        """
        display: Pygame display surface
        current_image: carla.Image (RGB)
        current_lidar: carla.LidarMeasurement
        camera_k: Intrinsic Matrix (3x3)
        """
        # 1. RENDER CAMERA (Left Side)
        if current_image:
            # Convert raw CARLA image to Numpy array
            array = np.frombuffer(current_image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (current_image.height, current_image.width, 4))
            array = array[:, :, :3]  # Remove Alpha
            array = array[:, :, ::-1]  # BGR -> RGB

            # Create Pygame surface
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            # --- DRAW BOUNDING BOXES ON CAMERA ---
            self._draw_bounding_boxes(surface, vehicle, all_actors, camera_k)

            # Blit to left side of screen
            display.blit(surface, (0, 0))

        # 2. RENDER LIDAR (Right Side)
        if current_lidar:
            self._render_lidar(display, current_lidar, vehicle, all_actors)

    def _render_lidar(self, display, lidar_data, ego_vehicle, actors):
        # 1. Init Surface
        self.lidar_surface.fill((0, 0, 0))

        # Scale: 5 pixels per meter (Zoomed out to ~80m width)
        lidar_scale = 5.0
        offset_x = self.half_w // 2
        offset_y = self.height // 2

        # 2. Render LiDAR Points (Green)
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))

        # Transform: Sensor X (Forward) -> Screen Up (-Y)
        # Transform: Sensor Y (Right)   -> Screen Right (+X)
        px = (points[:, 1] * lidar_scale + offset_x).astype(int)
        py = (-points[:, 0] * lidar_scale + offset_y).astype(int)

        # Fast drawing (Clip to screen)
        valid = (px >= 0) & (px < self.half_w) & (py >= 0) & (py < self.height)
        for x, y in zip(px[valid], py[valid]):
            self.lidar_surface.set_at((x, y), (0, 255, 0))

        # 3. Draw Red Bounding Boxes
        world_to_ego = np.array(ego_vehicle.get_transform().get_inverse_matrix())

        for target in actors:
            if target.id == ego_vehicle.id or 'vehicle' not in target.type_id:
                continue

            # --- CORRECTION START ---
            # We must use the bounding box LOCATION (offset) and EXTENT
            bbox = target.bounding_box
            c = bbox.location  # The center of the box relative to the car
            e = bbox.extent  # Half-width, half-length, half-height

            # Calculate the 4 bottom corners manually relative to the actor
            # We add 'c' to account for the offset you saw!
            local_corners = [
                carla.Location(x=c.x + e.x, y=c.y + e.y, z=c.z - e.z),  # Front Right
                carla.Location(x=c.x - e.x, y=c.y + e.y, z=c.z - e.z),  # Rear Right
                carla.Location(x=c.x - e.x, y=c.y - e.y, z=c.z - e.z),  # Rear Left
                carla.Location(x=c.x + e.x, y=c.y - e.y, z=c.z - e.z)  # Front Left
            ]
            # --- CORRECTION END ---

            # Transform and Draw
            screen_points = []
            target_transform = target.get_transform()

            for corner in local_corners:
                # Local -> World
                target_transform.transform(corner)
                # World -> Ego
                p_ego = self._apply_transform(corner, world_to_ego)

                # Ego -> Screen
                screen_x = int(p_ego[1] * lidar_scale + offset_x)
                screen_y = int(-p_ego[0] * lidar_scale + offset_y)
                screen_points.append((screen_x, screen_y))

            if len(screen_points) == 4:
                pygame.draw.lines(self.lidar_surface, (255, 0, 0), True, screen_points, 2)

        display.blit(self.lidar_surface, (self.half_w, 0))

    def _draw_bounding_boxes(self, surface, ego_vehicle, actors, K):
        """Draws 3D wireframe boxes on the camera image."""
        world_to_camera = np.array(ego_vehicle.get_transform().get_inverse_matrix())

        # UE4 (x-fwd, y-right, z-up) -> Camera (x-right, y-down, z-fwd)
        ue_to_cam = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        for npc in actors:
            if npc.id == ego_vehicle.id or "vehicle" not in npc.type_id:
                continue

            # 1. Get the 8 corners of the bounding box in World Space
            bb = npc.bounding_box
            verts = [v for v in bb.get_world_vertices(npc.get_transform())]

            # 2. Project vertices to Camera Space
            points_2d = []
            for v in verts:
                # World -> Camera Actor (Relative)
                p_loc = self._apply_transform(v, world_to_camera)

                # Account for Camera Height manually
                p_loc[2] -= 1.7  ### <--- SIMPLY SUBTRACT THE CAMERA HEIGHT HERE

                # Check if in front of camera (z > 0 in camera coords, but x > 0 in UE relative)
                if p_loc[0] < 0:
                    continue  # Behind camera

                # UE Relative -> Camera Optical (Kitti standard)
                p_cam = np.dot(ue_to_cam, np.append(p_loc, 1))[:3]

                # Camera Optical -> Image Pixels
                u, v, z = np.dot(K, p_cam)
                if z == 0: continue
                u, v = int(u / z), int(v / z)
                points_2d.append((u, v))

            # 3. Draw Lines if we have 8 points
            if len(points_2d) == 8:
                # Base
                pygame.draw.line(surface, (255, 0, 0), points_2d[0], points_2d[1], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[1], points_2d[2], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[2], points_2d[3], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[3], points_2d[0], 2)
                # Top
                pygame.draw.line(surface, (255, 0, 0), points_2d[4], points_2d[5], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[5], points_2d[6], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[6], points_2d[7], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[7], points_2d[4], 2)
                # Pillars
                pygame.draw.line(surface, (255, 0, 0), points_2d[0], points_2d[4], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[1], points_2d[5], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[2], points_2d[6], 2)
                pygame.draw.line(surface, (255, 0, 0), points_2d[3], points_2d[7], 2)

    def _apply_transform(self, location, transform_matrix):
        point = np.array([location.x, location.y, location.z, 1])
        point_transformed = np.dot(transform_matrix, point)
        return point_transformed[:3]

    @staticmethod
    def handle_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

# ==============================================================================
# -- setup methods --------------------------------------------------------------
# ==============================================================================
def clear_all_vehicles(client, world, logger):
    """
    Destroys all vehicles currently in the simulation to ensure a clean slate.
    """
    logger.info("Clearing all existing vehicles...")
    actor_list = world.get_actors()
    vehicle_list = actor_list.filter('vehicle.*')

    if not vehicle_list:
        logger.info("No vehicles found to clear.")
        return

    batch = [carla.command.DestroyActor(x) for x in vehicle_list]
    responses = client.apply_batch_sync(batch, True)

    # Optional: Check for errors
    errors = sum(1 for r in responses if r.error)
    if errors:
        logger.warning(f"{errors} errors occurred while clearing vehicles.")
    else:
        logger.info(f"Cleared {len(vehicle_list)} vehicles.")

def setup_carla(logger, client: carla):
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    logger.info(f"Client: {client_version} - Server: {server_version}")

    client.set_timeout(10.0)
    traffic_manager = client.get_trafficmanager()  # this is crashing
    sim_world = client.get_world()

    # -----
    # Synchronisation configuration
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    sim_world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)

def random_spawn(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    return world.try_spawn_actor(blueprint, spawn_point)

def setup_vehicle(world):
    logger = logging.getLogger()

    # Get a random blueprint.
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find('vehicle.tesla.model3')

    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    # Spawn point selection
    spawn_attempts = 0
    actor = random_spawn(world, blueprint)
    while actor is None and spawn_attempts < 20:
        actor = random_spawn(world, blueprint)
        spawn_attempts += 1
    if actor is None:
        logger.info("Could not spawn actor in 20 attempts")
        raise

    physics_control = actor.get_physics_control()
    physics_control.use_sweep_wheel_collision = True
    actor.apply_physics_control(physics_control)
    return actor


def spawn_traffic(client, world, amount=60):
    logger = logging.getLogger()

    # 1. Setup Traffic Manager
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(True)  # Match your simulation setting

    # 2. Get Blueprints
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    # Filter out Heavy vehicles and Bikes (Strict car filter)
    # We check the Blueprint ID for keywords indicating non-cars
    forbidden_keywords = ['truck', 'van', 'bus', 'sprinter', 'firetruck', 'ambulance', 'carlamotors']
    cars_only = []
    for bp in blueprints:
        bp_id = bp.id.lower()
        if not any(keyword in bp_id for keyword in forbidden_keywords):
            cars_only.append(bp)
    blueprints = cars_only
    logger.info(f"Blueprint library filtered to {len(blueprints)} car types (excluding trucks/vans).")

    # 3. Get Spawn Points
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if amount > number_of_spawn_points:
        amount = number_of_spawn_points

    # 4. Spawn Vehicles
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= amount:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # "Autopilot" means the Traffic Manager controls it
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(carla.command.SpawnActor(blueprint, transform)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))

    # 5. Execute Batch
    results = client.apply_batch_sync(batch, True)

    # 6. Return vehicles IDs
    vehicles_id_list = []
    for response in results:
        if not response.error:
            vehicles_id_list.append(response.actor_id)
    logger.info(f"Spawned {len(vehicles_id_list)} NPC vehicles.")
    return vehicles_id_list

# ==============================================================================
# -- Agent() --------------------------------------------------------------
# ==============================================================================
def set_random_destination(world, agent):
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
def main():
    # -----
    # Parsing input arguments
    # -----
    logger = getLogger(__name__)
    camera_width = 1242  # Standard KITTI width
    camera_height = 375  # Standard KITTI height
    max_lidar_points = 120000

    # -----
    # Setting up
    # -----
    logger.info("carla_env.Client setup started")
    carla_client = carla.Client('localhost', 2000)
    setup_carla(logger=logger, client=carla_client)
    logger.info("Carla Client started setup finished")

    kitti_gen = KittiGenerator(base_path="kitti_dataset", width=camera_width, height=camera_height, fov=90.0)
    # We use queues to ensure we get the data that matches the exact tick
    image_queue = queue.Queue()
    lidar_queue = queue.Queue()

    # -----
    # Carla Setup
    # -----
    logger.info("Setting up carla")
    world = carla_client.get_world()
    clear_all_vehicles(carla_client, world, logger)

    vehicle = setup_vehicle(world=world)

    # 1. Camera
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(camera_width))
    cam_bp.set_attribute('image_size_y', str(camera_height))
    cam_bp.set_attribute('fov', '90')
    # Spawn relative to vehicle
    cam_transform = carla.Transform(carla.Location(x=0.0, z=1.7))
    camera_sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    camera_sensor.listen(image_queue.put)

    # 2. LiDAR
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '1300000')
    lidar_bp.set_attribute('rotation_frequency', '10')  # 10Hz to match standard sim tick
    lidar_bp.set_attribute('range', '50')
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.7))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_queue.put)

    npc_ids = spawn_traffic(carla_client, world, amount=50)

    world.tick()

    logger.info("Setting up cameras")
    # Pygame
    pygame.init()
    pygame.font.init()
    hud_width = camera_width * 2  # double width for camera + LiDAR
    hud_height = camera_height * 2
    display = pygame.display.set_mode(
        (hud_width, hud_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("CARLA Simulation")
    hud = MinimalHUD(hud_width, hud_height)

    # 3) Agent
    logger.info("Setting up agent")
    agent = BasicAgent(vehicle=vehicle, target_speed=30)
    logger.info(f"Setup finished for vehicle at {vehicle.get_location()}")

    # 4) Simulation
    simstep = 0
    end_simulation = False

    logger.info("Starting simulation")
    set_random_destination(world, agent)   # fixme, required when running in no-controller mode
    try:
        while simstep < 300000:
            # Performing full run
            while not agent.done():
                # 1. Tick the world
                world.tick()

                # 2. Retrieve Sensor Data (Blocking get ensures sync)
                try:
                    # Get data generated by this tick
                    current_image = image_queue.get(timeout=2.0)
                    current_lidar = lidar_queue.get(timeout=2.0)
                except queue.Empty:
                    logger.warning("Missed a sensor frame!")
                    continue

                # 3. Save Data (Every X frames if you want to reduce size, currently every frame)
                if simstep % 3 == 0:
                    all_actors = world.get_actors(npc_ids)  # Pass NPCs for label generation
                    kitti_gen.save_frame(current_image, current_lidar, vehicle, all_actors)

                # ... (Keep your HUD, Agent Control, and Pygame logic) ...
                if MinimalHUD.handle_pygame_events():
                    end_simulation = True
                    break

                control = agent.run_step()
                control.manual_gear_shift = False
                vehicle.apply_control(control)

                # HUD overlay
                hud.tick()
                hud.render(
                    display=display,
                    current_image=current_image,
                    current_lidar=current_lidar,
                    camera_sensor=camera_sensor,
                    vehicle=vehicle,
                    all_actors=all_actors,
                    camera_k=kitti_gen.intrinsic  # Pass the K matrix
                )
                pygame.display.flip()
                simstep += 1

            set_random_destination(world, agent)
            logger.info("Destination reached")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        # -----
        # Cleaning up
        # -----
        logger.info("Closing down ..")
        pygame.quit()
        # FIX: Use correct variable names
        if 'camera_sensor' in locals() and camera_sensor:
            camera_sensor.stop()
            camera_sensor.destroy()
        if 'lidar_sensor' in locals() and lidar_sensor:
            lidar_sensor.stop()
            lidar_sensor.destroy()
        if 'vehicle' in locals() and vehicle:
            vehicle.destroy()
        if 'npc_ids' in locals() and npc_ids:
            carla_client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        logger.info("Exitting ..")

if __name__ == '__main__':
    main()
