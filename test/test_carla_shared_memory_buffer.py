from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper

FILENAME = "/dev/shm/test_carla_shared.dat"

class TestCarlaBuffer:
    def test_shared_memory_image(self):
        # ----- Setup -----
        width, height = 320, 240
        img_array = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)

        # Create a simple fake carla.Image object with .raw_data
        fake_carla_img = SimpleNamespace(
            width=width,
            height=height,
            raw_data=img_array.tobytes()
        )

        carla_buffer = CarlaWrapper(filename=FILENAME)

        # ----- Execute -----
        carla_buffer.write_image(image=fake_carla_img)

        # ----- Read back -----
        read_back = carla_buffer.read_images()
        read_back = carla_buffer.read_images()
        img_array2 = np.array(read_back).reshape(height, width, 4)

        # ----- Assert -----
        assert img_array2.shape == img_array.shape
        assert np.allclose(img_array2, img_array, atol=1), "Image roundtrip mismatch"