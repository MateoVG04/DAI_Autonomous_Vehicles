import carla
import argparse
import logging
import random
import os
import queue


# ==============================================================================
# -- Main Function -------------------------------------------------------------
# ==============================================================================

def main():
    """
    Main function to connect to CARLA, populate the world with traffic,
    and collect synchronized sensor data robustly, with frame skipping.
    """

    # --- Argument parsing ---
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--frames', default=100, type=int, help='Number of synchronized frames to collect')
    argparser.add_argument('--num-vehicles', default=70, type=int, help='Number of vehicles to spawn in traffic')
    argparser.add_argument('--num-walkers', default=40, type=int, help='Number of pedestrians to spawn')
    # *** NEW: Argument to control frame skipping for data diversity ***
    argparser.add_argument('--save-every', default=100, type=int, help='Save a frame every N frames.')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    actor_list = []

    # --- Client and World Setup ---
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        blueprint_library = world.get_blueprint_library()

        # ======================================================================
        # == 1. SPAWN THE EGO VEHICLE ==========================================
        # ======================================================================
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f'Spawned ego vehicle: {ego_vehicle.type_id}')

        # ======================================================================
        # == 2. POPULATE THE WORLD WITH TRAFFIC (Identical to your code) =======
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
            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                actor_list.append(vehicle)
                vehicle.set_autopilot(True, traffic_manager.get_port())
                vehicles_spawned += 1
        print(f"Successfully spawned {vehicles_spawned} vehicles.")

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
        print(f"Successfully spawned and started {walkers_spawned} walkers.")

        # ======================================================================
        # == 3. SET RANDOM WEATHER (Identical to your code) ====================
        # ======================================================================
        weather_presets = [
            carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon,
            carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon,
            carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset,
            carla.WeatherParameters.WetCloudySunset
        ]
        world.set_weather(random.choice(weather_presets))
        print(f"Set weather to: {world.get_weather()}")

        # ======================================================================
        # == 4. SETUP SENSORS ON EGO VEHICLE (Identical to your code) ==========
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
        # == 5. MAIN DATA COLLECTION LOOP (MODIFIED) ===========================
        # ======================================================================
        print(f'\nCollecting {args.frames} synchronized frames, saving one every {args.save_every} frames...')

        latest_images = {}
        frames_collected = 0
        while frames_collected < args.frames:
            world.wait_for_tick()
            try:
                while not image_queue.empty():
                    sensor_id, image = image_queue.get(block=False)
                    if image.frame not in latest_images: latest_images[image.frame] = {}
                    latest_images[image.frame][sensor_id] = image

                for frame, images in list(latest_images.items()):
                    if 'rgb' in images and 'semantic' in images:
                        rgb_img = images['rgb']

                        # *** MODIFICATION: Only save the frame if it meets the skip-rate condition ***
                        if rgb_img.frame % args.save_every == 0:
                            sem_img = images['semantic']

                            try:
                                # Save the synchronized pair to the local _out directory
                                rgb_img.save_to_disk(f'/test/rgb/{rgb_img.frame:06d}.png')
                                #sem_img.save_to_disk(f'/out/semantic/{sem_img.frame:06d}.png',
                                                     #carla_env.ColorConverter.CityScapesPalette)

                                # Only increment the counter AFTER a successful save
                                frames_collected += 1
                                print(f'Saved synchronized frame {sem_img.frame} ({frames_collected}/{args.frames})')

                            except Exception as e:
                                print(f'Could not save image {frame}. Error: {e}')

                        # Always remove the frame from the dictionary to save memory
                        del latest_images[frame]
            except queue.Empty:
                continue

    finally:
        # --- Cleanup ---
        print('\nCleaning up spawned actors...')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('Cleanup complete. Exiting.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nScript interrupted by user.')
    except Exception as e:
        logging.exception(e)