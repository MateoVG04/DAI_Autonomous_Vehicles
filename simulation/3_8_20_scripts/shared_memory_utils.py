# carla_writer_dat.py
import carla
import numpy as np
import os
import time

# --- Shared memory setup ---
MAX_ACTORS = 100
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3

# File sizes
pos_shape = (MAX_ACTORS, 3)
img_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
pos_size = np.prod(pos_shape) * 8  # float64
img_size = np.prod(img_shape)      # uint8
FILENAME = "/home/shared/carla_shared.dat"

total_size = pos_size + img_size

# Create file if it doesn't exist
if not os.path.exists(FILENAME):
    with open(FILENAME, "wb") as f:
        f.write(b"\x00" * total_size)

# Memory map
mm = np.memmap(FILENAME, dtype=np.uint8, mode="r+", shape=(total_size,))

# Helper views
pos_mm = np.ndarray(pos_shape, dtype=np.float64, buffer=mm[:pos_size].view(np.float64))
img_mm = np.ndarray(img_shape, dtype=np.uint8, buffer=mm[pos_size:])

# --- Connect to CARLA ---
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()

# Add a camera sensor
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
camera_bp.set_attribute('fov', '90')

spawn_point = world.get_map().get_spawn_points()[0]
camera = world.spawn_actor(camera_bp, spawn_point)

# Callback for camera images
def save_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    np.copyto(img_mm, array)

camera.listen(save_image)

# --- Main loop ---
try:
    while True:
        actors = world.get_actors()[:MAX_ACTORS]
        pos_mm[:] = 0  # reset
        for i, actor in enumerate(actors):
            loc = actor.get_location()
            pos_mm[i] = [loc.x, loc.y, loc.z]
        mm.flush()  # ensure reader sees updates
        time.sleep(0.05)
finally:
    camera.stop()
    camera.destroy()
    del mm
import os
import numpy as np
from typing import Tuple


class SharedMemoryManager:
    def __init__(self, filename: str,
                 image_shape: Tuple[int, int, int],
                 max_actors: int = 100,
                 ):
        """

        :param filename:
        :param image_shape: tuple of (image_height, image_width, image_channels)
        :param max_actors:
        """
        # ----- Configuring sizes
        # Position
        self.pos_shape = (max_actors, 3) # 3 for (x, y, z) coordinate
        self.max_actors = max_actors
        # Image
        self.image_shape = image_shape
        self.image_height, self.image_width, self.image_channels = image_shape

        # File sizes
        # todo add documentation for the datatype (see if we're correct)
        pos_size = np.prod(self.pos_shape) * 8  # 8 for float64
        img_size = np.prod(image_shape)  # 1 for uint8
        total_size = pos_size + img_size

        # ----- Creating binds for use in this class
        # Making sure the file already exists
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.write(b"\x00" * total_size)

        # Memory map
        self._mm = np.memmap(filename, dtype=np.uint8, mode="r+", shape=(total_size,))

        # Helper views
        # One for each type of data
        self._pos_mm = np.ndarray(self.pos_shape, dtype=np.float64, buffer=self._mm[:pos_size].view(np.float64))
        self._img_mm = np.ndarray(self.image_shape, dtype=np.uint8, buffer=self._mm[pos_size:])

    def __del__(self):
        """
        Destructor: Removing the bindings to the file (but not deleting the file!)
        """
        try:
            del self._mm
            del self._pos_mm
            del self._img_mm
        except AttributeError:
            # In case the attributes are already deleted
            pass

    def write_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(self.image_shape)
        np.copyto(self._img_mm, array)
        self._mm.flush()
        return

    def write_position(self, loc, index: int):
        """
        :param loc: Location fetched by `actor.get_location()` for example
        :param index: Index in the array to write
        """
        if index < 0 or index <= self.max_actors:
            raise
        self._pos_mm[0] = [loc.x, loc.y, loc.z]
        self._mm.flush()
        return
