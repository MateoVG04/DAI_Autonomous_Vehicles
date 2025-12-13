import weakref

import carla
import numpy as np

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager:
    """ Class for camera management"""

    def __init__(self,
                 client,
                 world,
                 parent_actor,
                 camera_width: int, camera_height: int,
                 shared_memory: CarlaWrapper):
        """Constructor method"""
        self.sensors = []
        self.surface = None
        self._parent = parent_actor
        self.shared_memory = shared_memory

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        blueprint_library = world.get_blueprint_library()

        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', str(camera_width))
        rgb_camera_bp.set_attribute('image_size_y', str(camera_height))
        self.sensors.append(world.spawn_actor(rgb_camera_bp, camera_transform, attach_to=self._parent))

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensors[0].listen(lambda image: CameraManager._parse_image(weak_self, image))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        """
        Function is to be used as a callback
        > self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        :param weak_self:
        :param image:
        :return:
        """
        self = weak_self()
        if not self:
            return
        self.shared_memory.write_image(image=image)


class LiDARManager:
    """
    Minimal LiDAR manager.
    """

    def __init__(
        self,
        client,
        world,
        parent_actor,
        shared_memory,
        range_m: float = 50.0,
        channels: int = 32,
        points_per_second: int = 56000,
        rotation_frequency: float = 10.0,
    ):
        self._parent = parent_actor
        self.shared_memory = shared_memory

        self.sensor = None
        self.latest_points = None  # np.ndarray (N, 4)
        self.frame = None

        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('range', str(range_m))
        lidar_bp.set_attribute('channels', str(channels))
        lidar_bp.set_attribute('points_per_second', str(points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(rotation_frequency))

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, z=2.5)
        )

        self.sensor = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self._parent
        )

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda data: LiDARManager._parse_lidar(weak_self, data)
        )

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

    @staticmethod
    def _parse_lidar(weak_self, data: carla.LidarMeasurement):
        self = weak_self()
        if not self:
            return

        # Each point: x, y, z, intensity
        self.latest_points = data.raw_data
        self.latest_points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.shared_memory.write_lidar_points(self.latest_points)