import carla
import argparse
import logging
import random
import os
import queue
import time

# ==============================================================================
# -- Map Definitions for Data Splitting ----------------------------------------
# ==============================================================================
# Define which maps to use for each dataset split. This ensures the model
# is validated and tested on environments it has never seen during training.
# Make sure these maps are available in your CARLA installation.
TRAIN_MAPS = []
VAL_MAPS = ['Town02']
TEST_MAPS = []


# ==============================================================================
# -- Data Collection Function --------------------------------------------------
# ==============================================================================

def collect_data_for_map(args, client, map_name, output_base_dir, split_name):
    """
    Loads a specific map, populates it, and collects synchronized sensor data.
    This function is designed to be called repeatedly for different maps.
    """
    logging.info(f"--- Starting data collection for map: {map_name} [{split_name} set] ---")
    actor_list = []

    try:
        # --- Load the new map ---
        world = client.load_world(map_name)
        # Let the world settle after a map change
        world.wait_for_tick()

        blueprint_library = world.get_blueprint_library()

        # --- Create output directories ---
        split_dir = os.path.join(output_base_dir, split_name)
        rgb_dir = os.path.join(split_dir, 'rgb')
        semantic_dir = os.path.join(split_dir, 'semantic')
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(semantic_dir, exist_ok=True)
        logging.info(f"Outputting data to: {split_dir}")

        # ======================================================================
        # == 1. SPAWN THE EGO VEHICLE ==========================================
        # ======================================================================
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        if not ego_vehicle:
            logging.error("Could not spawn ego vehicle. Skipping map.")
            return
        actor_list.append(ego_vehicle)
        logging.info(f'Spawned ego vehicle: {ego_vehicle.type_id}')

        # ======================================================================
        # == 2. POPULATE THE WORLD WITH TRAFFIC ================================
        # ======================================================================
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_blueprints = [bp for bp in blueprint_library.filter('vehicle.*') if
                              int(bp.get_attribute('number_of_wheels')) == 4]

        vehicles_spawned = 0
        for transform in spawn_points:
            if vehicles_spawned >= args.num_vehicles: break
            bp = random.choice(vehicle_blueprints)
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                actor_list.append(vehicle)
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles_spawned += 1
        logging.info(f"Successfully spawned {vehicles_spawned} vehicles.")

        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        walkers_spawned = 0
        for i in range(args.num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                walker = world.try_spawn_actor(random.choice(walker_blueprints), spawn_point)
                if walker is not None:
                    controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    if controller is not None:
                        actor_list.append(walker)
                        actor_list.append(controller)
                        controller.start()
                        controller.go_to_location(world.get_random_location_from_navigation())
                        walkers_spawned += 1
        logging.info(f"Successfully spawned {walkers_spawned} walkers.")

        # ======================================================================
        # == 3. SET RANDOM WEATHER =============================================
        # ======================================================================
        weather_presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon,
                           carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon,
                           carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset]
        world.set_weather(random.choice(weather_presets))
        logging.info(f"Set weather to: {world.get_weather()}")

        world.wait_for_tick()  # Allow weather to take effect

        # ======================================================================
        # == 4. SETUP SENSORS ON EGO VEHICLE ===================================
        # ======================================================================
        image_queue = queue.Queue()
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', '800');
        rgb_camera_bp.set_attribute('image_size_y', '600')
        rgb_camera = world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(rgb_camera)
        rgb_camera.listen(lambda image: image_queue.put(('rgb', image)))

        semantic_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_camera_bp.set_attribute('image_size_x', '800');
        semantic_camera_bp.set_attribute('image_size_y', '600')
        semantic_camera = world.spawn_actor(semantic_camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(semantic_camera)
        semantic_camera.listen(lambda image: image_queue.put(('semantic', image)))

        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        # ======================================================================
        # == 5. MAIN DATA COLLECTION LOOP ======================================
        # ======================================================================
        logging.info(
            f'Collecting {args.frames_per_map} synchronized frames, saving one every {args.save_every} frames...')

        latest_images = {}
        frames_collected = 0
        while frames_collected < args.frames_per_map:
            world.wait_for_tick()
            try:
                while not image_queue.empty():
                    sensor_id, image = image_queue.get(block=False)
                    if image.frame not in latest_images: latest_images[image.frame] = {}
                    latest_images[image.frame][sensor_id] = image

                for frame, images in list(latest_images.items()):
                    if 'rgb' in images and 'semantic' in images:
                        rgb_img = images['rgb']

                        if rgb_img.frame % args.save_every == 0:
                            sem_img = images['semantic']

                            rgb_path = os.path.join(rgb_dir, f'{map_name}_{rgb_img.frame:06d}.png')
                            sem_path = os.path.join(semantic_dir, f'{map_name}_{sem_img.frame:06d}.png')

                            # Save the synchronized pair
                            rgb_img.save_to_disk(rgb_path)
                            # *** CRITICAL: Save as RAW so the class IDs are preserved for training ***
                            sem_img.save_to_disk(sem_path, carla.ColorConverter.Raw)

                            frames_collected += 1
                            print(f'\rSaved {frames_collected}/{args.frames_per_map} frames for map {map_name}', end='')

                        del latest_images[frame]
            except queue.Empty:
                continue

    finally:
        # --- Cleanup for this map ---
        logging.info(f'\nCleaning up actors for map: {map_name}...')
        if actor_list:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        time.sleep(2)  # Give server time to clean up
        logging.info('Cleanup complete.')


# ==============================================================================
# -- Main Pipeline Function ----------------------------------------------------
# ==============================================================================

def main_pipeline():
    """
    Main function to orchestrate the entire data collection pipeline across
    multiple maps and dataset splits.
    """
    # --- Argument parsing ---
    argparser = argparse.ArgumentParser(description="CARLA Data Collection Pipeline")
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--frames-per-map', default=5000, type=int,
                           help='Number of synchronized frames to collect per map')
    argparser.add_argument('--num-vehicles', default=70, type=int, help='Number of vehicles to spawn in traffic')
    argparser.add_argument('--num-walkers', default=40, type=int, help='Number of pedestrians to spawn')
    argparser.add_argument('--save-every', default=50, type=int,
                           help='Save a frame every N simulation frames to increase data diversity.')
    # *** NEW: Argument for the base output directory ***
    argparser.add_argument('--output-dir', default='/mnt/data/carla_dataset',
                           help='Base directory to save the collected data.')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    map_splits = {
        'train': TRAIN_MAPS,
        'val': VAL_MAPS,
        'test': TEST_MAPS
    }

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)  # Increase timeout for map loading

        for split_name, maps in map_splits.items():
            if not maps:
                logging.warning(f"No maps defined for the '{split_name}' set. Skipping.")
                continue

            for map_name in maps:
                collect_data_for_map(args, client, map_name, args.output_dir, split_name)

        logging.info("--- Entire data collection pipeline finished successfully! ---")

    except Exception as e:
        logging.critical("An error occurred in the main pipeline.", exc_info=True)


if __name__ == '__main__':
    main_pipeline()