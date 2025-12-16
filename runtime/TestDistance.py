import carla
import random
import time
import queue
import numpy as np
import cv2
import logging

# --- CONFIGURATION ---
TOWN_MAP = 'Town04'
NUM_VEHICLES = 50
CAM_WIDTH = 800
CAM_HEIGHT = 600
FOV = 90.0
MODEL_PATH = "/home/shared/3_12_jupyter/bin/simulation/Model/unet_multiclass.pth"

# Make sure this import matches your actual file structure!
# If you saved the previous class as 'perception.py', change this.
try:
    from runtime.Distance import DistanceSystem
except ImportError:
    # Fallback if you are running from a different folder
    from Distance import DistanceSystem


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # 1. Initialize Perception System
    print("Initializing Perception AI...")

    # CHANGE 1: Remove 'cpu' to allow GPU usage if available.
    # RL needs high FPS.
    import torch
    device = 'cpu'
    print(f"Running inference on: {device}")

    perception = DistanceSystem(MODEL_PATH, CAM_WIDTH, CAM_HEIGHT, FOV, device=device)

    # 2. Setup CARLA Client
    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    # Load World
    if client.get_world().get_map().name.split('/')[-1] != TOWN_MAP:
        print(f"Loading {TOWN_MAP}...")
        world = client.load_world(TOWN_MAP)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 3. Spawn Traffic
    # CHANGE 2: Randomize Port to prevent "Bind Error" on restart
    tm_port = random.randint(4000, 6000)
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)

    traffic_actors = spawn_traffic(world, NUM_VEHICLES, traffic_manager)

    # 4. Spawn Hero Vehicle
    bp_lib = world.get_blueprint_library()
    hero_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    hero_bp.set_attribute('role_name', 'hero')

    spawn_points = world.get_map().get_spawn_points()
    hero_vehicle = None

    for _ in range(10):
        transform = random.choice(spawn_points)
        hero_vehicle = world.try_spawn_actor(hero_bp, transform)
        if hero_vehicle: break

    if not hero_vehicle:
        print("Failed to spawn hero vehicle.")
        return

    # Turn on Autopilot (Link to the random port)
    hero_vehicle.set_autopilot(True, tm_port)

    # 5. Setup Sensors
    sensor_queue = queue.Queue()

    # A. RGB Camera
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(CAM_WIDTH))
    cam_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
    cam_bp.set_attribute('fov', str(FOV))

    # Mount at dashboard (Must match LidarManager calibration!)
    cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))

    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=hero_vehicle)
    camera.listen(lambda data: sensor_queue.put(('rgb', data)))

    # B. Lidar
    lid_bp = bp_lib.find('sensor.lidar.ray_cast')
    lid_bp.set_attribute('range', '50')
    lid_bp.set_attribute('channels', '64')
    lid_bp.set_attribute('points_per_second', '1300000')
    lid_bp.set_attribute('rotation_frequency', '20')
    lid_bp.set_attribute('upper_fov', '2.0')
    lid_bp.set_attribute('lower_fov', '-25.0')
    lid_bp.set_attribute('dropoff_general_rate', '0.0')

    # Mount exactly same as camera
    lidar = world.spawn_actor(lid_bp, cam_transform, attach_to=hero_vehicle)
    lidar.listen(lambda data: sensor_queue.put(('lidar', data)))

    print("Simulation Running. Press 'q' to quit.")

    try:
        while True:
            world.tick()

            # Retrieve Sensor Data
            w_frame = world.get_snapshot().frame
            rgb_data = None
            lidar_data = None

            # Drain queue for current frame
            for _ in range(2):
                try:
                    name, data = sensor_queue.get(True, 1.0)
                    if name == 'rgb':
                        rgb_data = data
                    elif name == 'lidar':
                        lidar_data = data
                except queue.Empty:
                    print("Sensor Lag...")
                    continue

            if rgb_data is None or lidar_data is None:
                continue

            # --- PROCESS DATA ---

            # 1. Convert Image
            img_array = np.frombuffer(rgb_data.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (CAM_HEIGHT, CAM_WIDTH, 4))
            image_bgr = img_array[:, :, :3]

            # 2. Get Raw Lidar Bytes
            # We pass raw bytes because 'DistanceSystem' handles the projection
            lidar_bytes = lidar_data.raw_data

            # 3. Compute
            distance, dashboard = perception.compute(image_bgr, lidar_bytes)

            # 4. Display
            # Optional: Add text overlay on the main window
            cv2.putText(dashboard, f"DIST: {distance:.1f}m", (CAM_WIDTH + 20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Direct Simulation Debug", dashboard)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up...")
        if camera: camera.stop(); camera.destroy()
        if lidar: lidar.stop(); lidar.destroy()
        if hero_vehicle: hero_vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        cv2.destroyAllWindows()
        print("Done.")


def spawn_traffic(world, n_vehicles, tm):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()

    if n_vehicles > len(spawn_points): n_vehicles = len(spawn_points)
    random.shuffle(spawn_points)

    traffic = []
    for i in range(n_vehicles):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        bp.set_attribute('role_name', 'autopilot')

        vehicle = world.try_spawn_actor(bp, spawn_points[i])
        if vehicle:
            # Use the random port we created earlier
            vehicle.set_autopilot(True, tm.get_port())
            traffic.append(vehicle)

    print(f"Spawned {len(traffic)} background vehicles.")
    return traffic


if __name__ == "__main__":
    main()