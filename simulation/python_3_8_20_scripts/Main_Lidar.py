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


# ==============================================================================
# -- Shared Memory Classes (Included directly to fix NameError) ----------------
# ==============================================================================

class SharedMemoryArray:
    def __init__(self, data_shape: List[int], reserved_count: int, datatype) -> None:
        self.data_shape = data_shape
        self.reserved_count = reserved_count
        self.datatype = datatype
        self.datatype_size = {np.uint8: 1, np.float64: 8}[datatype]
        self.current_write_index = 0

    @property
    def slot_size(self):
        return np.prod(self.data_shape) * self.datatype_size

    @property
    def reserved_size(self):
        return self.slot_size * self.reserved_count


class SharedMemoryManager:
    def __init__(self, filename: str, data_arrays: List[SharedMemoryArray]):
        self.filepath_str: str = filename
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
        shared_array = self.data_arrays[buffer_index]
        slot_size = int(np.prod(shared_array.data_shape) * shared_array.datatype_size)
        index_array = len(self.data_arrays)
        return index_array + start_of_array + slot_index * slot_size

    def clear(self):
        self._mm[:] = 0
        self._mm.flush()

    def __del__(self):
        try:
            del self._mm
        except AttributeError:
            pass

    def init_file(self, filepath: Path):
        # Delete existing file to prevent size mismatches if config changes
        # if os.path.exists(filepath):
        #     os.remove(filepath)

        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)
                os.chmod(filepath, 0o666)
            return

        with open(filepath, "ab") as f:
            current_size = os.path.getsize(filepath)
            if self.total_size > current_size:
                f.write(b"\x00" * (self.total_size - current_size))

    def current_index(self, shared_array_index: int) -> int:
        return int(self._write_index_mm[shared_array_index])

    def set_write_index(self, shared_array_index: int, index_value: int):
        self._write_index_mm[shared_array_index] = index_value
        self._mm.flush()

    def increment_write_index(self, shared_array_index: int):
        next_index = self.current_index(shared_array_index) + 1
        if next_index == self.data_arrays[shared_array_index].reserved_count:
            next_index = 0
        self.set_write_index(shared_array_index=shared_array_index, index_value=next_index)

    def write_data(self, shared_array_index: int, input_data: np.ndarray):
        self.write_data_at(shared_array_index=shared_array_index,
                           write_index=self.current_index(shared_array_index),
                           input_data=input_data)
        self.increment_write_index(shared_array_index=shared_array_index)

    def write_data_at(self, shared_array_index: int, write_index: int, input_data: np.ndarray):
        shared_array = self.data_arrays[shared_array_index]
        array = np.frombuffer(input_data, dtype=shared_array.datatype).ravel()
        start = self.write_offset(buffer_index=shared_array_index, slot_index=write_index)
        end = start + self.data_arrays[shared_array_index].slot_size
        self._mm[start:end] = array
        self._mm.flush()


class CarlaWrapper:
    class CarlaDataType(IntEnum):
        images = 0
        object_detected = 1
        waypoint = 3

    def __init__(self, filename, image_width: int, image_height: int):
        data_arrays = [
            SharedMemoryArray(data_shape=[image_height, image_width, 3],
                              reserved_count=100, datatype=np.uint8),
            SharedMemoryArray(data_shape=[image_height, image_width, 3],
                              reserved_count=100, datatype=np.uint8),
            SharedMemoryArray(data_shape=[33],
                              reserved_count=100, datatype=np.float64),
        ]
        self.shared_memory = SharedMemoryManager(filename=filename, data_arrays=data_arrays)

    def clear(self):
        self.shared_memory.clear()

    # We reuse this method for Lidar data
    def write_object_detected(self, image: np.ndarray):
        # Ensure array is contiguous and uint8 before writing
        array = np.ascontiguousarray(image, dtype=np.uint8)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.object_detected.value, input_data=array)


# ==============================================================================
# -- Lidar Manager -------------------------------------------------------------
# ==============================================================================
class LidarManager:
    def __init__(self, world, parent_actor, image_width, image_height, carla_wrapper):
        self.world = world
        self.parent = parent_actor
        self.width = image_width
        self.height = image_height
        self.shm_wrapper = carla_wrapper

        self.lidar_range = 40.0
        self.pixels_per_meter = self.height / (self.lidar_range * 2)
        self.sensor = None

        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '100000')
        bp.set_attribute('upper_fov', '15.0')
        bp.set_attribute('lower_fov', '-25.0')
        bp.set_attribute('dropoff_general_rate', '0.0')

        transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        self.sensor = self.world.spawn_actor(bp, transform, attach_to=self.parent)
        self.sensor.listen(self._callback)

    def _callback(self, data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            lidar_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            lidar_x = points[:, 0]
            lidar_y = points[:, 1]
            lidar_z = points[:, 2]

            mask = (np.abs(lidar_x) <= self.lidar_range) & (np.abs(lidar_y) <= self.lidar_range)
            lidar_x = lidar_x[mask]
            lidar_y = lidar_y[mask]
            lidar_z = lidar_z[mask]

            u = (self.width / 2) + lidar_y * self.pixels_per_meter
            v = (self.height / 2) - lidar_x * self.pixels_per_meter

            u = np.clip(u, 0, self.width - 1).astype(int)
            v = np.clip(v, 0, self.height - 1).astype(int)

            lidar_img[v, u, 1] = 255  # Green
            lidar_img[v, u, 2] = np.clip((lidar_z + 2.0) * 50, 0, 255).astype(np.uint8)  # Blue/Height

            # Write using the wrapper's method
            self.shm_wrapper.write_object_detected(lidar_img)

        except Exception as e:
            print(f"Lidar Error: {e}")

    def stop(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


# ==============================================================================
# -- Main ----------------------------------------------------------------------
# ==============================================================================
def setup_vehicle(world):
    bp_lib = world.get_blueprint_library()
    # Try finding model3, fallback to random if not found
    bp = bp_lib.find('vehicle.tesla.model3')
    if not bp:
        bp = random.choice(bp_lib.filter('vehicle.*'))

    bp.set_attribute('role_name', 'hero')
    spawn_points = world.get_map().get_spawn_points()
    return world.spawn_actor(bp, random.choice(spawn_points))


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    camera_width = 800
    camera_height = 600

    # 1. Connect
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 2. Settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # 3. Spawn
    vehicle = setup_vehicle(world)
    logger.info("Vehicle spawned")

    # 4. Setup Shared Memory
    shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"

    # Ensure directory exists if using /dev/shm it should be fine,
    # but good practice for other paths
    if not os.path.exists(os.path.dirname(shm_lidar_path)):
        # On strict linux systems /dev/shm exists.
        pass

    shm_lidar_wrapper = CarlaWrapper(filename=shm_lidar_path,
                                     image_width=camera_width,
                                     image_height=camera_height)
    shm_lidar_wrapper.clear()

    # 5. Start Lidar
    lidar_manager = LidarManager(world=world,
                                 parent_actor=vehicle,
                                 image_width=camera_width,
                                 image_height=camera_height,
                                 carla_wrapper=shm_lidar_wrapper)

    logger.info("Simulation running with Lidar...")

    try:
        while True:
            world.tick()
            vehicle.set_autopilot(True)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        lidar_manager.stop()
        if vehicle:
            vehicle.destroy()
        logger.info("Cleaned up.")


if __name__ == '__main__':
    main()