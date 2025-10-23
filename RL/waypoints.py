import carla
import numpy as np
import math
import cv2
import time
import random

from agents.navigation.global_route_planner import GlobalRoutePlanner
from waypoints_utils import *


# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

try:
    world = client.get_world()
    print(f"Connected to CARLA on port {2000}")
except Exception as e:
    raise RuntimeError("Cannot connect to CARLA. Make sure server is running.") from e

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)

# Spawn ego vehicle
blueprint_library = world.get_blueprint_library()
ego_vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_points = world.get_map().get_spawn_points()

ego_vehicle = None
for sp in spawn_points:
    try:
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, sp)
        print("Vehicle spawned at:", sp)
        break
    except RuntimeError:
        continue

if ego_vehicle is None:
    raise RuntimeError("No free spawn points available")

# Spawn vehicle in front (not working)
spawn_transform = compute_safe_spawn_location_ahead(world, ego_vehicle, 20)

front_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

front_vehicle = world.try_spawn_actor(front_vehicle_bp, spawn_transform)

if front_vehicle is None:
    print("Failed to spawn front vehicle — try a different distance or location.")
else:
    print("Front vehicle spawned successfully!")

# Attach camera to vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Define camera offset (behind and above the vehicle)
distance_behind = 8.0
height_above = 3.0

# Define spectator
spectator = world.get_spectator()
vehicle_transform = ego_vehicle.get_transform()
spectator_transform = update_spectator(vehicle_transform, distance_behind, height_above)
spectator.set_transform(spectator_transform)

# Setup radar
latest_distance_ahead = 50.0  # default max distance

radar_bp = blueprint_library.find('sensor.other.radar')
radar_bp.set_attribute('horizontal_fov', '30')
radar_bp.set_attribute('vertical_fov', '5')
radar_bp.set_attribute('range', '50')

radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=ego_vehicle)

def radar_callback(data):
    global latest_distance_ahead
    # Keep only objects roughly in front (±5° horizontal, ±2° vertical)
    forward_detections = [d for d in data if abs(math.degrees(d.azimuth)) < 5 and abs(math.degrees(d.altitude)) < 2]
    if forward_detections:
        latest_distance_ahead = min(d.depth for d in forward_detections)
    else:
        latest_distance_ahead = 50.0  # max distance

radar_sensor.listen(radar_callback)


# Setup Global Route Planner
planner = GlobalRoutePlanner(world.get_map(), sampling_resolution=SAMPLING_RESOLUTION)
start_location = spawn_points[0].location
end_location = spawn_points[-1].location
route = planner.trace_route(start_location, end_location)
waypoints = [wp for wp, _ in route]

# Enable autopilot
ego_vehicle.set_autopilot(True)

# Main Loop: compute state vector
try:
    while True:
        world.tick()

        # Update ego
        ego_pos, ego_yaw = get_ego_transform(ego_vehicle)

        # Closest waypoint
        distances = [np.linalg.norm(np.array([wp.transform.location.x, wp.transform.location.y]) - ego_pos[:2])
                     for wp in waypoints]
        closest_idx = int(np.argmin(distances))

        # Ego speed
        velocity = ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # Ego acceleration
        acc = ego_vehicle.get_acceleration()
        acceleration = math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

        # Distance to car ahead
        distance_to_front = latest_distance_ahead

        # State vector
        state_waypoints = waypoints[closest_idx:closest_idx + WAYPOINT_FRAME_SIZE]
        state_vector = build_state_vector(ego_pos, ego_yaw, state_waypoints)

        # Combined state
        state = np.concatenate([state_vector, [speed, acceleration, distance_to_front]], axis=0)
        print("State:", state)

        # Update spectator to follow vehicle
        vehicle_transform = ego_vehicle.get_transform()
        spectator_transform = update_spectator(vehicle_transform, distance_behind, height_above)
        spectator.set_transform(spectator_transform)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting simulation.")

finally:
    # Cleanup
    camera.stop()
    camera.destroy()
    radar_sensor.stop()
    radar_sensor.destroy()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    cv2.destroyAllWindows()
    print("Actors destroyed, simulation ended.")
