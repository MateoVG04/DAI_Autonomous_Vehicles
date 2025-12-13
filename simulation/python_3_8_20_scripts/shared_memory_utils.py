from enum import IntEnum
from pathlib import Path
from typing import List

import carla
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


class SharedMemoryArray:
    def __init__(self,
                 data_shape: List[int],
                 reserved_count: int,
                 datatype) -> None:
        """
        A definition for a type of data saved in the buffer file

        :param data_shape: Data shape to be converted into a linear size (12, 12, 1) -> 144 for example
        :param reserved_count: How many of those objects can be placed
        :param datatype: Size also depends on the datatype
        """
        self.data_shape = data_shape
        self.reserved_count = reserved_count
        self.datatype = datatype

        self.datatype_size = {np.uint8: 1, np.float32: 4, np.float64: 8}[datatype]

        self.current_write_index = 0

    @property
    def slot_size(self):
        return np.prod(self.data_shape) * self.datatype_size

    @property
    def reserved_size(self):
        """
        Size to be reserved in the buffer (in bytes)
        :return:
        """
        return self.slot_size * self.reserved_count

class SharedMemoryManager:
    def __init__(self, filename: str,
                 data_arrays: List[SharedMemoryArray],
                 ):
        """

        :param filename:
        """
        self.filepath_str: str = filename

        # ----- Data Arrays
        self.write_array = SharedMemoryArray(data_shape=[1], reserved_count=len(data_arrays), datatype=np.uint8)
        self.data_arrays = data_arrays

        # ----- Creating binds for use in this class
        # Making sure the file already exists
        path = Path(filename)
        self.init_file(path)

        # ---- Helper views
        # Main memory
        # | write_indices | ... all memory arrays |
        self._mm = np.memmap(filename, dtype=np.uint8, mode="r+", shape=(self.total_size,))
        # Write indices
        self._write_index_mm = np.ndarray(len(self.data_arrays), dtype=np.uint8, buffer=self._mm[:self.write_array.reserved_size])

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
        """
        Destructor: Removing the bindings to the file (but not deleting the file!)
        """
        try:
            del self._mm
        except AttributeError:
            # In case the attributes are already deleted
            pass

    def init_file(self, filepath: Path):
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)
                os.chmod(filepath, 0o666)
            return

        # Checking if size is correct
        with open(filepath, "ab") as f:  # append mode
            current_size = os.path.getsize(filepath)
            if self.total_size > current_size:
                f.write(b"\x00" * (self.total_size - current_size))
    # -----
    # write_index operations
    # -----
    def current_index(self, shared_array_index: int) -> int:
        return int(self._write_index_mm[shared_array_index])

    def set_write_index(self, shared_array_index: int, index_value: int):
        self._write_index_mm[shared_array_index] = index_value
        self._mm.flush()

    def increment_write_index(self, shared_array_index: int):
        next_index = self.current_index(shared_array_index)+1
        if next_index == self.data_arrays[shared_array_index].reserved_count:
            next_index = 0 # This makes the write_index circular
        self.set_write_index(shared_array_index=shared_array_index, index_value=next_index)

    # -----
    # Write operations
    # -----
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

    # -----
    # Read operations
    # -----
    def read_data_array(self, shared_array_index: int) -> np.ndarray:
        # Return a snapshot (copy) to avoid referencing live memory
        shared_array = self.data_arrays[shared_array_index]
        starting_pos = self.write_offset(buffer_index=shared_array_index, slot_index=0)

        shape = [shared_array.reserved_count, *shared_array.data_shape]
        return np.copy(np.ndarray(
            shape=shape,
                       dtype=shared_array.datatype,
                       buffer=self._mm[starting_pos:starting_pos + shared_array.reserved_size].view(
                           shared_array.datatype)
                       ))

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
        lidar_points = 2

    def __init__(self, filename, image_width: int, image_height: int, max_lidar_points: int):
        data_arrays = [
            SharedMemoryArray(data_shape=[image_height, image_width, 3], # Raw images
                              reserved_count=100,
                              datatype=np.uint8),
            SharedMemoryArray(data_shape=[image_height, image_width, 3], # Object Detected images
                              reserved_count=100,
                              datatype=np.uint8),
            SharedMemoryArray(data_shape=[33], # 33 for now
                                reserved_count=100,
                                datatype=np.float64),
            SharedMemoryArray(data_shape=[max_lidar_points, 4],  # LiDAR points x, y, z, intensity
                              reserved_count=100,
                              datatype=np.float32),
        ]
        self.shared_memory = SharedMemoryManager(filename=filename,
                                                 data_arrays=data_arrays)

        self.max_lidar_points = max_lidar_points

    def clear(self):
        self.shared_memory.clear()

    # ----- Raw Images
    def write_image(self, image):
        # image.save_to_disk(f'/home/s0203301/project/images/{image.frame:08d}.png')

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # drop alpha
        array = np.ascontiguousarray(array)

        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.images.value, input_data=array)
        return

    def read_images(self) -> np.ndarray:
        return self.shared_memory.read_data_array(shared_array_index=self.CarlaDataType.images.value)

    def read_image(self, index: int) -> np.ndarray:
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=index)

    @property
    def latest_image_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.images.value)

    def read_latest_image(self) -> np.ndarray:
        slot_index = self.latest_image_index -1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.images.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=slot_index)

    # ----- Object Detected
    @property
    def object_detected_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.object_detected.value)

    def read_latest_object_detected(self) -> np.ndarray:
        slot_index = self.latest_image_index -1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.object_detected.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.object_detected.value, slot_index=slot_index)

    def write_object_detected(self, image: np.ndarray):
        array = np.frombuffer(image, dtype=np.uint8)
        array = np.ascontiguousarray(array)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.object_detected.value, input_data=array)
        return

    # ----- Waypoints
    def write_waypoint(self, waypoint: np.ndarray):
        array = np.frombuffer(waypoint, dtype=np.float64)
        array = np.ascontiguousarray(array)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.waypoint.value, input_data=array)
        return

    def read_waypoints(self) -> np.ndarray:
        return self.shared_memory.read_data_array(shared_array_index=self.CarlaDataType.waypoint.value)

    # ----- LiDAR
    @property
    def latest_lidar_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.lidar_points.value)

    def write_lidar_points(self, points: np.ndarray):
        """
        Write a LiDAR frame to shared memory.
        Points shape: (N,4) with x,y,z,intensity
        Will truncate if N > MAX_LIDAR_POINTS
        """
        points = np.ascontiguousarray(points[:self.max_lidar_points], dtype=np.float32)

        # Pad with zeros if less points than max
        if points.shape[0] < self.max_lidar_points:
            pad = np.zeros((self.max_lidar_points - points.shape[0], 4), dtype=np.float32)
            points = np.vstack((points, pad))

        self.shared_memory.write_data(
            shared_array_index=self.CarlaDataType.lidar_points.value,
            input_data=points
        )

    def read_latest_lidar_points(self) -> np.ndarray:
        slot_index =  0 # self.latest_lidar_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.lidar_points.value].reserved_count - 1

        # read raw 1D array
        points_flat = self.shared_memory.read_data(
            shared_array_index=self.CarlaDataType.lidar_points.value,
            slot_index=slot_index
        )

        # reshape to 2D (N,4)
        points = points_flat.reshape(-1, 4)
        return points