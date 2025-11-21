import Pyro4
import Pyro4.naming
import threading
import carla
import random
import sys
import logging

from carlaEnvironment import CarlaEnv


# Start Name Server in background thread

def nameserver():
    Pyro4.naming.startNSloop(host="0.0.0.0", port=9090)

ns_thread = threading.Thread(target=nameserver, daemon=True)
ns_thread.start()

print("Embedded Pyro4 Name Server started on port 9090")


# CARLA setup

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

# Register CARLA env with Name Server
daemon = Pyro4.Daemon(host="0.0.0.0")   # remote access supported
ns = Pyro4.locateNS(host="localhost", port=9090)

uri = daemon.register(env, objectId="carla.environment")
ns.register("carla.environment", uri)

print("Carla environment registered with Name Server")
print("URI:", uri)

print("CARLA Pyro server is now running...")
daemon.requestLoop()
