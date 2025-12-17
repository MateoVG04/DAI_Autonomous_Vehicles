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
        self.front_camera = None
        self.chase_camera = None
        self._parent = parent_actor
        self.shared_memory = shared_memory

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        CHASE_CAM_TRANSFORM = carla.Transform(
            carla.Location(x=-6.5, z=2.8),
            carla.Rotation(pitch=-12.0)
        )
        blueprint_library = world.get_blueprint_library()

        rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', str(camera_width))
        rgb_camera_bp.set_attribute('image_size_y', str(camera_height))

        self.front_camera = world.spawn_actor(
            rgb_camera_bp,
            camera_transform,
            attach_to=self._parent
        )
        self.sensors.append(self.front_camera)

        # Chase camera (NIEUW)
        self.chase_camera = world.spawn_actor(
            rgb_camera_bp,
            CHASE_CAM_TRANSFORM,
            attach_to=self._parent
        )
        self.sensors.append(self.chase_camera)
        self.chase_camera.listen(self._on_chase_image)


        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensors[0].listen(lambda image: CameraManager._parse_image(weak_self, image))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def _on_chase_image(self, image):
        self.shared_memory.write_chase_image(image)

    def destroy(self):
        for sensor in self.sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        self.front_camera = None
        self.chase_camera = None

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
        points_per_second: int = 100000,
        rotation_frequency: float = 20.0,
        z_offset: float = 1.73
    ):
        self._parent = parent_actor
        self.shared_memory = shared_memory

        self.sensor = None
        self.latest_points = None  # np.ndarray (N, 4)
        self.frame = None

        blueprint_library = world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1300000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '2.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')


        lidar_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7)
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