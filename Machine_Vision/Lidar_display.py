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
        # Open in Read-Write (r+) mode
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
        except:
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
            SharedMemoryArray(data_shape=[image_height, image_width, 3], reserved_count=100, datatype=np.uint8),
            SharedMemoryArray(data_shape=[image_height, image_width, 3], reserved_count=100, datatype=np.uint8),
            SharedMemoryArray(data_shape=[33], reserved_count=100, datatype=np.float64),
        ]
        self.shared_memory = SharedMemoryManager(filename=filename, data_arrays=data_arrays)

    @property
    def latest_image_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.images.value)

    def read_latest_image(self) -> np.ndarray:
        slot_index = self.latest_image_index - 1
        if slot_index == -1: slot_index = 99
        return self.shared_memory.read_data(self.CarlaDataType.images.value, slot_index)

    @property
    def object_detected_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.object_detected.value)

    def read_latest_object_detected(self) -> np.ndarray:
        slot_index = self.object_detected_index - 1
        if slot_index == -1: slot_index = 99
        return self.shared_memory.read_data(self.CarlaDataType.object_detected.value, slot_index)


# ==============================================================================
# -- Viewer --------------------------------------------------------------------
# ==============================================================================
def main():
    print("Initializing Viewer...")
    width, height = 800, 600

    shm_camera = "/dev/shm/carla_shared_Rune.dat"
    shm_lidar = "/dev/shm/carla_shared_Rune_Lidar.dat"

    while not os.path.exists(shm_camera) or not os.path.exists(shm_lidar):
        print("Waiting for simulation memory...")
        time.sleep(1)

    cam_wrapper = CarlaWrapper(shm_camera, width, height)
    lid_wrapper = CarlaWrapper(shm_lidar, width, height)

    print("Running. Press 'q' to exit.")

    while True:
        try:
            # 1. Read Camera
            # The simulation now writes BGR. OpenCV expects BGR. No conversion needed.
            img_bgr = cam_wrapper.read_latest_image()
            img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)

            # 2. Read Lidar
            img_lidar = lid_wrapper.read_latest_object_detected()
            img_lidar = np.ascontiguousarray(img_lidar, dtype=np.uint8)

            # Marker
            cx, cy = width // 2, height // 2
            cv2.arrowedLine(img_lidar, (cx, cy + 10), (cx, cy - 20), (0, 0, 255), 2)

            cv2.imshow("RGB Camera", img_bgr)
            cv2.imshow("Lidar BEV", img_lidar)

        except Exception:
            pass

        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()