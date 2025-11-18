import Pyro4
import carla
import random
from carlaEnvironment import CarlaEnv

def setup():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    print(f"Spawned vehicle {vehicle.type_id} at {spawn_point.location}")

    # ---- THIRD-PERSON CHASE CAMERA ----
    spectator = world.get_spectator()

    def update_spectator(world_snapshot):
        transform = vehicle.get_transform()
        spectator_transform = carla.Transform(
            transform.location
            + transform.get_forward_vector() * -8
            + carla.Location(z=3),
            transform.rotation
        )
        spectator.set_transform(spectator_transform)

    world.on_tick(update_spectator)
    print("Third-person chase camera active.")

    return world, vehicle

world, vehicle = setup()

env = CarlaEnv(world, vehicle)

deamon = Pyro4.Daemon()

uri = deamon.register(env, objectId="carla.environment")

print("Carla env registered with uri:", uri)
print("Carla env is ready")

deamon.requestLoop()
