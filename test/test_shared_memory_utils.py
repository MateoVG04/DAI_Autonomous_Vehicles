from io import BytesIO

import numpy as np
from PIL import Image

from simulation.python_3_8_20_scripts.shared_memory_utils import SharedMemoryArray, SharedMemoryManager

FILENAME = "/dev/shm/test_carla_shared.dat"

class TestSharedMemoryBuffer:
    def test_shared_memory_setup(self):
        # -----
        # Setup
        data_arrays = [
            SharedMemoryArray(data_shape=[1],
                              reserved_count=100,
                              datatype=np.uint8),
        ]
        testing_value = np.ndarray(shape=(1,1), dtype=np.uint8)
        testing_value[0] = 123

        # -----
        # Execute
        shared_memory = SharedMemoryManager(filename=FILENAME, data_arrays=data_arrays)
        shared_memory.clear()

        shared_memory.write_data_at(shared_array_index=0, input_data=testing_value, write_index=0)
        data_array = shared_memory.read_data_array(shared_array_index=0)

        # -----
        # Assert
        assert len(data_array) != 0
        assert data_array[0] == testing_value[0]

    def test_shared_memory_image(self):
        # -----
        # Setup
        width, height = 320, 240
        img_array = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)

        # Encode to bytes (simulate image being shared)
        pil_img = Image.fromarray(img_array)
        buf = BytesIO()
        pil_img = Image.fromarray(img_array)
        img_bytes = np.array(pil_img, dtype=np.uint8).ravel()
        buf.close()

        # Prepare shared memory to hold the bytes
        data_arrays = [
            SharedMemoryArray(data_shape=[height, width, 4],
                              reserved_count=1,
                              datatype=np.uint8),
        ]

        shared_memory = SharedMemoryManager(filename=FILENAME, data_arrays=data_arrays)
        shared_memory.clear()

        # ----- Execute -----
        shared_memory.write_data_at(shared_array_index=0, input_data=img_bytes, write_index=0)
        read_back_array = shared_memory.read_data_array(0)

        # ----- Assert -----
        assert np.allclose(read_back_array, img_array)

    def test_shared_memory_auto_indexing(self):
        # -----
        # Setup
        data_arrays = [
            SharedMemoryArray(data_shape=[1],
                              reserved_count=10,
                              datatype=np.uint8),
        ]
        testing_value = np.ndarray(shape=1, dtype=np.uint8)
        testing_value[0] = 100

        testing_value_two = np.ndarray(shape=1, dtype=np.uint8)
        testing_value_two[0] = 110

        testing_value_three = np.ndarray(shape=1, dtype=np.uint8)
        testing_value_three[0] = 111

        # -----
        # Execute
        shared_memory = SharedMemoryManager(filename=FILENAME, data_arrays=data_arrays)
        shared_memory.clear()

        index_no_write = shared_memory.current_index(shared_array_index=0)
        # Two writes
        shared_memory.write_data(shared_array_index=0, input_data=testing_value)
        shared_memory.write_data(shared_array_index=0, input_data=testing_value_two)

        #
        index_two_writes = shared_memory.current_index(shared_array_index=0)
        shared_memory.write_data(shared_array_index=0, input_data=testing_value_three)

        index_three_writes = shared_memory.current_index(shared_array_index=0)

        data_array = shared_memory.read_data_array(shared_array_index=0)

        # -----
        # Assert
        assert index_no_write == 0
        assert index_two_writes == 2
        assert index_three_writes == 3

        assert len(data_array) != 0

        assert data_array[0] == testing_value[0]
        assert data_array[1] == testing_value_two[0]
        assert data_array[2] == testing_value_three[0]