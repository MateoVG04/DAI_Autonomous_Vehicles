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
        shared_memory.write_data(shared_array_index=0, input_data=testing_value)
        data_array = shared_memory.read_data_array(shared_array_index=0)

        # -----
        # Assert
        assert data_array
        assert data_array[0] == testing_value[0]

    def test_shared_memory_image(self):
        # -----
        # Setup
        width, height = 320, 240
        img_array = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)

        # Encode to bytes (simulate image being shared)
        pil_img = Image.fromarray(img_array)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        # Prepare shared memory to hold the bytes
        data_arrays = [
            SharedMemoryArray(data_shape=[len(img_bytes)],
                              reserved_count=1,
                              datatype=np.uint8),
        ]

        shared_memory = SharedMemoryManager(filename=FILENAME, data_arrays=data_arrays)

        # ----- Execute -----
        shared_memory.write_data(shared_array_index=0, input_data=img_bytes)
        read_back = shared_memory.read_data_array(shared_array_index=0)

        # Decode back to image
        read_bytes = bytes(read_back)
        pil_img2 = Image.open(BytesIO(read_bytes))
        img_array2 = np.array(pil_img2)

        # ----- Assert -----
        assert img_array2.shape == img_array.shape
        assert np.allclose(img_array2, img_array, atol=1), "Image roundtrip mismatch"
