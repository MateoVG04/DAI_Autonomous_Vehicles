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

    @property
    def reserved_size(self):
        """
        Size to be reserved in the buffer (in bytes)
        :return:
        """
        return np.prod((*self.data_shape, self.reserved_count, self.datatype_size))


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
        self._write_index_mm = np.ndarray(1, dtype=np.uint8, buffer=self._mm[:self.write_array.reserved_size])
        # View for each type of shared buffer array
        self._mm_handles = []
        for cur_i, shared_array in enumerate(self.data_arrays):
            starting_pos =sum(val.reserved_size for val in self.data_arrays[:cur_i])
            self._mm_handles.append(
                np.ndarray(shared_array.data_shape,
                           dtype=shared_array.datatype,
                           buffer=self._mm[starting_pos:starting_pos+shared_array.reserved_size].view(shared_array.datatype)
                           )
            )

    @property
    def total_size(self):
        return sum(val.reserved_size for val in self.data_arrays)

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

    def write_data(self, shared_array_index: int, input_data: np.ndarray) -> None:
        shared_array = self.data_arrays[shared_array_index]
        array = np.frombuffer(input_data, dtype=shared_array.datatype)
        array = array.reshape(shared_array.data_shape)
        np.copyto(self._mm_handles[shared_array_index], array)
        self._mm.flush()
        return

    def read_data_array(self, shared_array_index: int) -> np.ndarray:
        # Return a snapshot (copy) to avoid referencing live memory
        return np.copy(self._mm_handles[shared_array_index])

class CarlaWrapper:
    class CarlaDataType(IntEnum):
        images = 1

    def __init__(self, filepath):
        data_arrays = [
            SharedMemoryArray(data_shape=[32, 32, 1],
                              reserved_count=100,
                              datatype=np.uint8),
        ]
        self.shared_memory = SharedMemoryManager(filename=filepath,
                                                 data_arrays=data_arrays)

    def write_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.images.value, input_data=array)
        return
