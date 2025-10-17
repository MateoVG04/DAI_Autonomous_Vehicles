from enum import IntEnum
from pathlib import Path
from typing import List
import numpy as np
import os

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

        self.datatype_size = {np.uint8: 1, np.float64: 8}[datatype]

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
        # ----- Data Arrays
        self.write_array = SharedMemoryArray(data_shape=[1], reserved_count=len(data_arrays), datatype=np.uint8)
        self.data_arrays = data_arrays

        # ----- Creating binds for use in this class
        # Making sure the file already exists
        path = Path(filename)
        if not os.path.exists(path):
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
        with open(filepath, "wb") as f:
            f.write(b"\x00" * self.total_size)

    def set_write_index(self, buffer_index: int, index_value: int):
        self._write_index_mm[buffer_index] = index_value
        self._mm.flush()

    def current_index(self, shared_array_index: int) -> int:
        return int(self._write_index_mm[shared_array_index])

    def write_data(self, shared_array_index: int, input_data: np.ndarray):
        self.write_data_at(shared_array_index=shared_array_index,
                           write_index=self.current_index(shared_array_index),
                           input_data=input_data)
        self.set_write_index(buffer_index=shared_array_index, index_value=self.current_index(shared_array_index)+1)

    def write_data_at(self, shared_array_index: int, write_index: int, input_data: np.ndarray):
        shared_array = self.data_arrays[shared_array_index]
        array = np.frombuffer(input_data, dtype=shared_array.datatype).ravel()
        start = self.write_offset(buffer_index=shared_array_index, slot_index=write_index)
        end = start + self.data_arrays[shared_array_index].slot_size
        self._mm[start:end] = array
        self._mm.flush()

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
        # Return a snapshot (copy) to avoid referencing live memory
        shared_array = self.data_arrays[shared_array_index]
        starting_pos = self.write_offset(buffer_index=shared_array_index, slot_index=slot_index)

        return np.copy(np.ndarray(
            shape=shared_array.data_shape,
                       dtype=shared_array.datatype,
                       buffer=self._mm[starting_pos:starting_pos + shared_array.slot_size].view(
                           shared_array.datatype)
                       ))

class CarlaWrapper:
    class CarlaDataType(IntEnum):
        images = 0

    def __init__(self, filename):
        data_arrays = [
            SharedMemoryArray(data_shape=[320, 240, 4],
                              reserved_count=100,
                              datatype=np.uint8),
        ]
        self.shared_memory = SharedMemoryManager(filename=filename,
                                                 data_arrays=data_arrays)

    def clear(self):
        self.shared_memory.clear()

    def write_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.images.value, input_data=array)
        return

    def read_images(self) -> np.ndarray:
        return self.shared_memory.read_data_array(shared_array_index=self.CarlaDataType.images.value)

    def read_image(self, index: int) -> np.ndarray:
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=index)
