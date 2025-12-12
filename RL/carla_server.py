import Pyro4
import Pyro4.naming
import threading
import carla
import logging
from carla_env import CarlaEnv

"""
Carla Pyro4 Server. Starts a Pyro4 Name Server and registers a CARLA environment for remote access.
"""

CARLA_TIMEOUT = 120.0  # seconds

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

def nameserver():
    """
    Start a Pyro4 Name Server to allow remote object lookup.
    :return: None
    """
    Pyro4.naming.startNSloop(host="0.0.0.0", port=9090)


def setup() -> carla.Client:
    """
    Setup CARLA client, load world, spawn vehicle, and configure 3rd person camera.
    :return: None
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(CARLA_TIMEOUT)

    return client


def spectator_update(env_instance: 'CarlaEnv', snapshot):
    """Updates the spectator camera to follow the vehicle currently held by the environment."""

    # CRITICAL: Safely check if the vehicle exists and is active
    if env_instance.ego_vehicle and env_instance.ego_vehicle.is_alive:
        spectator = env_instance.world.get_spectator()
        vehicle = env_instance.ego_vehicle

        # Calculate transform relative to the current vehicle
        transform = vehicle.get_transform()
        spectator_transform = carla.Transform(
            transform.location
            + transform.get_forward_vector() * -8  # 8 meters behind
            + carla.Location(z=3),  # 3 meters up
            transform.rotation
        )
        spectator.set_transform(spectator_transform)



if __name__ == "__main__":
    # Start Name Server in background thread
    ns_thread = threading.Thread(target=nameserver, daemon=True)
    ns_thread.start()
    logger.info( "Name Server started on port 9090")

    # Carla setup
    client = setup()
    env = CarlaEnv(client)
    world = client.get_world()

    world.on_tick(lambda snapshot: spectator_update(env, snapshot))

    # Register CARLA env with Name Server
    daemon = Pyro4.Daemon(host="0.0.0.0")  # remote access supported
    ns = Pyro4.locateNS(host="localhost", port=9090)
    uri = daemon.register(env, objectId="carla.environment")
    ns.register("carla.environment", uri)
    logger.info( "Carla environment registered with Name Server")

    logger.info( "CARLA Pyro server is now running...")
    daemon.requestLoop()
