import Pyro4
import Pyro4.naming
import threading
import carla
import random
import logging

from carlaEnvironment import CarlaEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

def nameserver():
    Pyro4.naming.startNSloop(host="0.0.0.0", port=9090)


def setup():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world("Town01")
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)

    # 3rd person camera
    spectator = world.get_spectator()

    def update_spectator(snapshot):
        transform = vehicle.get_transform()
        spectator_transform = carla.Transform(
            transform.location
            + transform.get_forward_vector() * -8
            + carla.Location(z=3),
            transform.rotation
        )
        spectator.set_transform(spectator_transform)

    world.on_tick(update_spectator)

    return world, vehicle


if __name__ == "__main__":
    # Start Name Server in background thread
    ns_thread = threading.Thread(target=nameserver, daemon=True)
    ns_thread.start()
    logger.log(logging.INFO, "Name Server started on port 9090")

    # Carla setup
    world, vehicle = setup()
    env = CarlaEnv(world, vehicle)

    # Register CARLA env with Name Server
    daemon = Pyro4.Daemon(host="0.0.0.0")  # remote access supported
    ns = Pyro4.locateNS(host="localhost", port=9090)
    uri = daemon.register(env, objectId="carla.environment")
    ns.register("carla.environment", uri)
    logger.log(logging.INFO, "Carla environment registered with Name Server")

    logger.log(logging.INFO, "CARLA Pyro server is now running...")
    daemon.requestLoop()
