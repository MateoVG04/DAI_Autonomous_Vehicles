import logging
import random
import sys
import numpy as np
import carla
import time


# --- Import your specific classes here ---
# from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
# (Or paste the classes provided in your prompt at the top of this file)

# ==============================================================================
# -- Lidar Manager (Adapted for your CarlaWrapper) -----------------------------
# ==============================================================================
class LidarManager:
    def __init__(self, world, parent_actor, image_width, image_height, carla_wrapper):
        self.world = world
        self.parent = parent_actor
        self.width = image_width
        self.height = image_height

        # We use your custom wrapper
        self.shm_wrapper = carla_wrapper

        # Lidar Settings
        self.lidar_range = 40.0
        self.pixels_per_meter = self.height / (self.lidar_range * 2)
        self.sensor = None

        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(self.lidar_range))
        bp.set_attribute('rotation_frequency', '20')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '100000')
        bp.set_attribute('upper_fov', '15.0')
        bp.set_attribute('lower_fov', '-25.0')
        bp.set_attribute('dropoff_general_rate', '0.0')

        transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        self.sensor = self.world.spawn_actor(bp, transform, attach_to=self.parent)
        self.sensor.listen(self._callback)

    def _callback(self, data):
        try:
            # 1. Convert raw Lidar data to Numpy
            points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))

            # 2. Prepare BEV Image
            lidar_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            lidar_x = points[:, 0]
            lidar_y = points[:, 1]
            lidar_z = points[:, 2]

            # 3. Filter and Project
            mask = (np.abs(lidar_x) <= self.lidar_range) & (np.abs(lidar_y) <= self.lidar_range)
            lidar_x = lidar_x[mask]
            lidar_y = lidar_y[mask]
            lidar_z = lidar_z[mask]

            u = (self.width / 2) + lidar_y * self.pixels_per_meter
            v = (self.height / 2) - lidar_x * self.pixels_per_meter

            u = np.clip(u, 0, self.width - 1).astype(int)
            v = np.clip(v, 0, self.height - 1).astype(int)

            # 4. Colorize
            lidar_img[v, u, 1] = 255  # Green
            lidar_img[v, u, 2] = np.clip((lidar_z + 2.0) * 50, 0, 255).astype(np.uint8)  # Blue (Height)

            # 5. WRITE USING YOUR WRAPPER
            # We use 'write_object_detected' because it accepts a numpy array
            self.shm_wrapper.write_object_detected(lidar_img)

        except Exception as e:
            print(f"Lidar Error: {e}")

    def stop(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


# ==============================================================================
# -- Helper & Main -------------------------------------------------------------
# ==============================================================================
def setup_vehicle(world):
    bp = random.choice(world.get_blueprint_library().filter('vehicle.tesla.model3'))
    bp.set_attribute('role_name', 'hero')
    spawn_points = world.get_map().get_spawn_points()
    return world.spawn_actor(bp, random.choice(spawn_points))


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configuration
    camera_width = 800
    camera_height = 600

    # Connect
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Spawn
    vehicle = setup_vehicle(world)
    logger.info("Vehicle spawned")

    # --- SHARED MEMORY SETUP ---

    # 1. Lidar Wrapper
    # We create a new file specifically for Lidar
    shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"
    shm_lidar_wrapper = CarlaWrapper(filename=shm_lidar_path,
                                     image_width=camera_width,
                                     image_height=camera_height)
    # Clear previous data
    shm_lidar_wrapper.clear()

    # 2. Lidar Manager
    lidar_manager = LidarManager(world=world,
                                 parent_actor=vehicle,
                                 image_width=camera_width,
                                 image_height=camera_height,
                                 carla_wrapper=shm_lidar_wrapper)

    logger.info("Simulation running...")

    try:
        while True:
            world.tick()

            # Simple autopilot for testing
            vehicle.set_autopilot(True)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        lidar_manager.stop()
        vehicle.destroy()
        logger.info("Cleaned up.")


if __name__ == '__main__':
    main()