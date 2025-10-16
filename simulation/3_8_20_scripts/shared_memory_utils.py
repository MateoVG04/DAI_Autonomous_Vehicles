from pathlib import Path
from typing import Tuple
import numpy as np
import os

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
        path = Path(filename)
        if not os.path.exists(path):
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
