import logging
import random
import sys
import time
import os
import mmap
from enum import IntEnum
from pathlib import Path
from typing import List
import numpy as np
import carla

# --- CONFIGURATION ---
RECORD_DATA = False  # Set False to stop filling disk while debugging
DATASET_PATH = "captured_data"
TOWN_MAP = 'Town04'  # Town04 is best for highway/lane testing
NUM_VEHICLES = 50  # Number of cars
TM_PORT = 8005  # <--- NEW: Specific port to avoid "Bind Error"


# ==============================================================================
# -- Shared Memory Classes -----------------------------------------------------
# ==============================================================================
class SharedMemoryArray:
    def __init__(self, data_shape: List[int], reserved_count: int, datatype) -> None:
        self.data_shape = data_shape
        self.reserved_count = reserved_count
        self.datatype = datatype
        self.datatype_size = {np.uint8: 1, np.float64: 8}[datatype]

    @property
    def slot_size(self): return np.prod(self.data_shape) * self.datatype_size

    @property
    def reserved_size(self): return self.slot_size * self.reserved_count


class SharedMemoryManager:
    def __init__(self, filename: str, data_arrays: List[SharedMemoryArray]):
        self.filepath_str = filename
        self.write_array = SharedMemoryArray(data_shape=[1], reserved_count=len(data_arrays), datatype=np.uint8)
        self.data_arrays = data_arrays
        path = Path(filename)
        self.init_file(path)
        self._mm = np.memmap(filename, dtype=np.uint8, mode="r+", shape=(self.total_size,))
        self._write_index_mm = np.ndarray(len(self.data_arrays), dtype=np.uint8,
                                          buffer=self._mm[:self.write_array.reserved_size])

    @property
    def total_size(self):
        return sum(val.reserved_size for val in self.data_arrays) + len(self.data_arrays)

    def write_offset(self, buffer_index: int, slot_index: int):
        start_of_array = sum(v.reserved_size for v in self.data_arrays[:buffer_index])
        slot_size = int(
            np.prod(self.data_arrays[buffer_index].data_shape) * self.data_arrays[buffer_index].datatype_size)
        return len(self.data_arrays) + start_of_array + slot_index * slot_size

    def clear(self):
        self._mm[:] = 0; self._mm.flush()

    def __del__(self):
        try:
            del self._mm
        except:
            pass

    def init_file(self, filepath: Path):
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(b"\x00" * self.total_size); os.chmod(filepath, 0o666)
        else:
            with open(filepath, "ab") as f:
                if self.total_size > os.path.getsize(filepath): f.write(
                    b"\x00" * (self.total_size - os.path.getsize(filepath)))

    def current_index(self, idx):
        return int(self._write_index_mm[idx])

    def set_write_index(self, idx, val):
        self._write_index_mm[idx] = val

    def increment_write_index(self, idx):
        next_i = self.current_index(idx) + 1
        if next_i == self.data_arrays[idx].reserved_count: next_i = 0
        self.set_write_index(idx, next_i)

    def write_data(self, idx, data):
        w_idx = self.current_index(idx)
        arr = np.frombuffer(data, dtype=self.data_arrays[idx].datatype).ravel()
        s = self.write_offset(idx, w_idx)
        e = s + self.data_arrays[idx].slot_size
        self._mm[s:e] = arr
        self.increment_write_index(idx)


class CarlaWrapper:
    class CarlaDataType(IntEnum): images = 0; object_detected = 1; waypoint = 3

    def __init__(self, filename, w, h):
        data = [
            SharedMemoryArray([h, w, 3], 100, np.uint8),
            SharedMemoryArray([h, w, 3], 100, np.uint8),
            SharedMemoryArray([33], 100, np.float64)
        ]
        self.shared_memory = SharedMemoryManager(filename, data)

    def clear(self): self.shared_memory.clear()

    def write_image(self, img):
        self.shared_memory.write_data(self.CarlaDataType.images.value, np.ascontiguousarray(img, dtype=np.uint8))

    def write_object_detected(self, img):
        self.shared_memory.write_data(self.CarlaDataType.object_detected.value,
                                      np.ascontiguousarray(img, dtype=np.uint8))


# ==============================================================================
# -- Sensor Managers -----------------------------------------------------------
# ==============================================================================

class LidarManager:
    def __init__(self, world, parent_actor, image_width, image_height, carla_wrapper):
        self.world = world
        self.parent = parent_actor
        self.width = image_width
        self.height = image_height
        self.shm_wrapper = carla_wrapper

        # CONFIG
        self.lidar_range = 50.0
        self.fov = 90.0

        # Calibration Matrix
        focal = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = focal
        self.K[0, 2] = self.width / 2.0
        self.K[1, 2] = self.height / 2.0

        self.sensor = None
        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('channels', '64')
        bp.set_attribute('points_per_second', '1300000')
        bp.set_attribute('upper_fov', '2.0')
        bp.set_attribute('lower_fov', '-25.0')
        bp.set_attribute('dropoff_general_rate', '0.0')

        # MOUNT POSITION: Must match Camera!
        self.sensor = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=self.parent)
        self.sensor.listen(self._callback)

    def _callback(self, data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            # 1. Filter Geometry
            lidar_x = points[:, 0]
            lidar_y = points[:, 1]
            lidar_z = points[:, 2]

            # Range & Ground Filter
            mask = (lidar_x > 0.5) & (lidar_x < self.lidar_range) & (lidar_z > -1.6)

            lidar_x = lidar_x[mask]
            lidar_y = lidar_y[mask]
            lidar_z = lidar_z[mask]

            # 2. Project to 2D Image
            u_coords = (lidar_y * self.K[0, 0] / lidar_x) + self.K[0, 2]
            v_coords = (-lidar_z * self.K[1, 1] / lidar_x) + self.K[1, 2]

            # 3. Screen Filter
            screen_mask = (u_coords >= 0) & (u_coords < self.width) & \
                          (v_coords >= 0) & (v_coords < self.height)

            u_coords = u_coords[screen_mask].astype(int)
            v_coords = v_coords[screen_mask].astype(int)
            depths = lidar_x[screen_mask]

            # 4. Create Depth Image
            depth_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Encode Depth: Bright = Close, Dark = Far
            norm_depth = 1.0 - (depths / self.lidar_range)
            intensity = (norm_depth * 255).astype(np.uint8)

            # Draw THICK circles
            for i in range(len(u_coords)):
                val = int(intensity[i])
                cv2.circle(depth_img, (u_coords[i], v_coords[i]), 3, (val, val, val), -1)

            self.shm_wrapper.write_object_detected(depth_img)

            if RECORD_DATA and data.frame % 10 == 0:
                import cv2
                fname = f"{DATASET_PATH}/lidar/{data.frame:08d}.png"
                cv2.imwrite(fname, depth_img)

        except Exception as e:
            pass

    def stop(self):
        if self.sensor: self.sensor.destroy()


class InternalCameraManager:
    def __init__(self, world, parent_actor, image_width, image_height, carla_wrapper):
        self.world = world
        self.parent = parent_actor
        self.width = image_width
        self.height = image_height
        self.shm_wrapper = carla_wrapper
        self.sensor = None
        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov', '90')
        bp.set_attribute('motion_blur_intensity', '0.45')

        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.sensor = self.world.spawn_actor(bp, transform, attach_to=self.parent)
        self.sensor.listen(self._callback)

    def _callback(self, data):
        try:
            array = np.frombuffer(data.raw_data, dtype=np.uint8)
            array = array.reshape((self.height, self.width, 4))
            bgr_image = array[:, :, :3]

            self.shm_wrapper.write_image(bgr_image)

            if RECORD_DATA and data.frame % 10 == 0:
                import cv2
                fname = f"{DATASET_PATH}/rgb/{data.frame:08d}.png"
                cv2.imwrite(fname, bgr_image)

        except Exception as e:
            print(f"Camera Error: {e}")

    def stop(self):
        if self.sensor: self.sensor.destroy()


# ==============================================================================
# -- TRAFFIC GENERATION --------------------------------------------------------
# ==============================================================================
def spawn_traffic(client, world, number_of_vehicles):
    logging.info(f"Spawning {number_of_vehicles} vehicles...")

    # --- FIX 1: Use specific port 8005 to avoid bind error ---
    traffic_manager = client.get_trafficmanager(TM_PORT)

    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.global_percentage_speed_difference(30.0)

    # --- FIX 2: Enable Hybrid Physics (Prevents cars from freezing) ---
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(50.0)  # Only calculate physics for cars near us

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    else:
        number_of_vehicles = number_of_spawn_points

    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]

    traffic_list = []

    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles: break
        blueprint = random.choice(vehicle_bps)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')

        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle:
            # --- FIX 3: Assign Autopilot to SPECIFIC TM PORT ---
            vehicle.set_autopilot(True, TM_PORT)
            traffic_list.append(vehicle)

    logging.info(f"Spawned {len(traffic_list)} vehicles.")
    return traffic_list


def set_random_weather(world):
    presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.ClearSunset,
    ]
    weather = random.choice(presets)
    world.set_weather(weather)
    logging.info(f"Weather set to random preset.")


# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if RECORD_DATA:
        os.makedirs(f"{DATASET_PATH}/rgb", exist_ok=True)
        os.makedirs(f"{DATASET_PATH}/lidar", exist_ok=True)

    camera_width = 800
    camera_height = 600

    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    # Load Map
    world = client.load_world(TOWN_MAP)
    set_random_weather(world)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 1. Spawn Traffic
    traffic_actors = spawn_traffic(client, world, number_of_vehicles=NUM_VEHICLES)

    # 2. Spawn Hero Vehicle
    bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
    bp.set_attribute('role_name', 'hero')
    spawn_points = world.get_map().get_spawn_points()

    hero_vehicle = None
    for _ in range(10):
        transform = random.choice(spawn_points)
        hero_vehicle = world.try_spawn_actor(bp, transform)
        if hero_vehicle: break

    if not hero_vehicle:
        logger.error("Could not spawn hero vehicle!")
        return

    logger.info("Hero Vehicle spawned")

    # --- FIX 4: Activate Hero Autopilot OUTSIDE Loop (Correct way) ---
    hero_vehicle.set_autopilot(True, TM_PORT)

    # --- SETUP SHARED MEMORY ---
    shm_camera_path = "/dev/shm/carla_shared_Rune.dat"
    shm_camera = CarlaWrapper(shm_camera_path, camera_width, camera_height)
    shm_camera.clear()

    shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"
    shm_lidar = CarlaWrapper(shm_lidar_path, camera_width, camera_height)
    shm_lidar.clear()

    # --- START SENSORS ---
    lidar_manager = LidarManager(world, hero_vehicle, camera_width, camera_height, shm_lidar)
    camera_manager = InternalCameraManager(world, hero_vehicle, camera_width, camera_height, shm_camera)

    logger.info("Sensors Active. Press Ctrl+C to stop.")

    try:
        while True:
            # Sync mode tick
            world.tick()

            # Weather change
            if world.get_snapshot().frame % 1000 == 0:
                set_random_weather(world)

    except KeyboardInterrupt:
        pass
    finally:
        lidar_manager.stop()
        camera_manager.stop()
        if hero_vehicle: hero_vehicle.destroy()

        logger.info("Destroying traffic...")
        client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        logger.info("Cleaned up.")


if __name__ == '__main__':
    main()