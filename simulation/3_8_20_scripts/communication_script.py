from datetime import datetime

import carla
import numpy as np
from shared_memory_utils import SharedMemoryManager

# --- Shared memory setup ---
FILENAME = "/dev/shm/carla_shared.dat"
MAX_ACTORS = 100
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 4

if __name__ == '__main__':
    shared_memory = SharedMemoryManager(filename=FILENAME, image_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), max_actors=MAX_ACTORS)

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', '90')

    spawn_point = world.get_map().get_spawn_points()[0]
    camera = world.spawn_actor(camera_bp, spawn_point)


    # Callback for camera images
    def camera_callback(image):
        shared_memory.write_image(image=image)
    camera.listen(camera_callback)

    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < 1:
        ...
    print("Gracefull exit!")