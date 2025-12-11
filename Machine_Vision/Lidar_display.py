import cv2
import time
import numpy as np
import os
import mmap
from enum import IntEnum
from pathlib import Path
from typing import List


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

        # Open in Read-Write mode (r+)
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

    def __del__(self):
        try:
            del self._mm
        except AttributeError:
            pass

    def init_file(self, filepath: Path):
        pass

    def current_index(self, shared_array_index: int) -> int:
        return int(self._write_index_mm[shared_array_index])

    def read_data(self, shared_array_index: int, slot_index: int):
        shared_array = self.data_arrays[shared_array_index]
        start = self.write_offset(buffer_index=shared_array_index, slot_index=slot_index)
        buf = self._mm[start:start + shared_array.slot_size]
        return np.frombuffer(buf, dtype=shared_array.datatype).reshape(shared_array.data_shape).copy()


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

    # --- Read Image (Slot 0) ---
    @property
    def latest_image_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.images.value)

    def read_latest_image(self) -> np.ndarray:
        slot_index = self.latest_image_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.images.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=slot_index)

    # --- Read Object/Lidar (Slot 1) ---
    @property
    def object_detected_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.object_detected.value)

    def read_latest_object_detected(self) -> np.ndarray:
        slot_index = self.object_detected_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.object_detected.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.object_detected.value,
                                            slot_index=slot_index)


# ==============================================================================
# -- Viewer Logic --------------------------------------------------------------
# ==============================================================================

def main():
    print("Initializing Viewer...")

    width = 800
    height = 600

    # 1. Define Paths
    shm_camera_path = "/dev/shm/carla_shared_Rune.dat"
    shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"

    # 2. Wait for Files
    while not os.path.exists(shm_camera_path) or not os.path.exists(shm_lidar_path):
        print("Waiting for simulation to create shared memory files...")
        time.sleep(1)

    # 3. Initialize Wrappers
    print("Connecting to Camera Memory...")
    camera_wrapper = CarlaWrapper(filename=shm_camera_path, image_width=width, image_height=height)

    print("Connecting to Lidar Memory...")
    lidar_wrapper = CarlaWrapper(filename=shm_lidar_path, image_width=width, image_height=height)

    print("Viewer Running. Press 'q' to exit.")

    while True:
        try:
            # --- Read Camera (RGB) ---
            rgb_img = camera_wrapper.read_latest_image()
            rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
            # Convert RGB to BGR for OpenCV display
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            # --- Read Lidar (BEV) ---
            lidar_img = lidar_wrapper.read_latest_object_detected()
            lidar_img = np.ascontiguousarray(lidar_img, dtype=np.uint8)

            # Draw Ego Arrow on Lidar
            cx, cy = width // 2, height // 2
            cv2.arrowedLine(lidar_img, (cx, cy + 10), (cx, cy - 20), (0, 0, 255), 2)

            # --- Display ---
            cv2.imshow("RGB Camera", bgr_img)
            cv2.imshow("Lidar BEV", lidar_img)

        except Exception as e:
            # pass on read errors (sync issues)
            pass

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()