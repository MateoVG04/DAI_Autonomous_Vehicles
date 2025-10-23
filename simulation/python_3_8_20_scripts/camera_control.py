import weakref

import carla

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
                 shared_memory_filepath: str):
        """Constructor method"""
        self.sensors = []
        self.surface = None
        self._parent = parent_actor
        self.shared_memory = CarlaWrapper(filename=shared_memory_filepath, image_width=camera_width, image_height=camera_height)

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
