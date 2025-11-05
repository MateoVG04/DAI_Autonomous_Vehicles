import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from waypoints_utils import *
import random
from agents.navigation.basic_agent import BasicAgent
import time

def setup_carla(client: carla):
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    print(f"Client: {client_version} - Server: {server_version}")

    client.set_timeout(10.0)
    sim_world = client.get_world()

    # -----
    # Synchronisation configuration
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    sim_world.apply_settings(settings)


def random_spawn(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    return world.try_spawn_actor(blueprint, spawn_point)

def setup_vehicle(world):

    # Get a random blueprint.
    blueprint_library = world.get_blueprint_library()
    blueprint = random.choice(blueprint_library.filter('vehicle.*'))
    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    # Spawn point selection
    spawn_attempts = 0
    actor = random_spawn(world, blueprint)
    while actor is None and spawn_attempts < 20:
        actor = random_spawn(world, blueprint)
        spawn_attempts += 1
    if actor is None:
        print("Could not spawn actor in 20 attempts")
        raise

    physics_control = actor.get_physics_control()
    physics_control.use_sweep_wheel_collision = True
    actor.apply_physics_control(physics_control)
    return actor

def set_random_destination(world, agent):
    spawn_points = world.get_map().get_spawn_points()
    destination = random.choice(spawn_points).location
    agent.set_destination(destination)


client = carla.Client('localhost', 2000)
setup_carla(client)
world = client.get_world()
vehicle = setup_vehicle(world)

agent = BasicAgent(vehicle=vehicle, target_speed=30)


# Spawn vehicle in front (not working)
spawn_transform = compute_safe_spawn_location_ahead(world, vehicle, 20)

# Attach camera to vehicle
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Define camera offset (behind and above the vehicle)
distance_behind = 8.0
height_above = 3.0

# Define spectator
spectator = world.get_spectator()
vehicle_transform = vehicle.get_transform()
spectator_transform = update_spectator(vehicle_transform, distance_behind, height_above)
spectator.set_transform(spectator_transform)

# Setup radar
latest_distance_ahead = 50.0  # default max distance

radar_bp = world.get_blueprint_library().find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '30')
radar_bp.set_attribute('vertical_fov', '5')
radar_bp.set_attribute('range', '50')

radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

def radar_callback(data):
    global latest_distance_ahead
    # Keep only objects roughly in front (±5° horizontal, ±2° vertical)
    forward_detections = [d for d in data if abs(math.degrees(d.azimuth)) < 5 and abs(math.degrees(d.altitude)) < 2]
    if forward_detections:
        latest_distance_ahead = min(d.depth for d in forward_detections)
    else:
        latest_distance_ahead = 50.0  # max distance

radar_sensor.listen(radar_callback)

set_random_destination(world, agent)

# # Setup Global Route Planner
# planner = GlobalRoutePlanner(world.get_map(), sampling_resolution=SAMPLING_RESOLUTION)
# start_location = spawn_points[0].location
# end_location = spawn_points[-1].location
# route = planner.trace_route(start_location, end_location)
# waypoints = [wp for wp, _ in route]


end_simulation = False

try:
    while not end_simulation:
        while not agent.done():
            world.tick()

            control = agent.run_step()
            control.manual_gear_shift = False
            vehicle.apply_control(control)

            # Ego speed
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            # Ego acceleration
            acc = vehicle.get_acceleration()
            acceleration = math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

            # Distance to car ahead
            distance_to_front = latest_distance_ahead

            # Update ego
            ego_pos, ego_yaw = get_ego_transform(vehicle)

            # State vector
            # state_vector = build_state_vector(ego_pos, ego_yaw, waypoints, 15)
            #
            # # Combined state
            # state = np.concatenate([state_vector, [speed, acceleration, distance_to_front]], axis=0)
            # print("State:", state)
            # waypoint_buffer.write_image(state)

            # Update spectator to follow vehicle
            vehicle_transform = vehicle.get_transform()
            spectator_transform = update_spectator(vehicle_transform, distance_behind, height_above)
            spectator.set_transform(spectator_transform)

# TODO: Add to wrapper for data writing

except KeyboardInterrupt:
    print("Exiting simulation.")

finally:
    # Cleanup
    camera.stop()
    camera.destroy()
    radar_sensor.stop()
    radar_sensor.destroy()
    if vehicle is not None:
        vehicle.destroy()
    print("Actors destroyed, simulation ended.")
