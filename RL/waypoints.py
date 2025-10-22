import carla
import numpy as np
import math
import cv2
import time
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ===============================
# Parameters
# ===============================
WAYPOINT_FRAME_SIZE = 15  # number of waypoints in the state vector
SAMPLING_RESOLUTION = 2.0  # meters between global waypoints
CARLA_HOST = 'localhost'
CARLA_PORT = 2000  # make sure this matches your running CARLA server

# ===============================
# Helper functions
# ===============================
def get_ego_transform(vehicle):
    """Returns ego vehicle position and yaw angle in radians."""
    transform = vehicle.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    return np.array([loc.x, loc.y, loc.z]), yaw

def global_to_local(ego_pos, ego_yaw, waypoint_pos):
    """Transforms a waypoint from global to ego-local coordinates."""
    dx = waypoint_pos[0] - ego_pos[0]
    dy = waypoint_pos[1] - ego_pos[1]
    x_local = math.cos(-ego_yaw) * dx - math.sin(-ego_yaw) * dy
    y_local = math.sin(-ego_yaw) * dx + math.cos(-ego_yaw) * dy
    return np.array([x_local, y_local])

def build_state_vector(ego_pos, ego_yaw, waypoints):
    """Build a FIFO list of local waypoints for the agent."""
    state = []
    for wp in waypoints[:WAYPOINT_FRAME_SIZE]:
        local_coords = global_to_local(ego_pos, ego_yaw, (wp.transform.location.x, wp.transform.location.y))
        state.append(local_coords[0])  # only lateral info
    while len(state) < WAYPOINT_FRAME_SIZE:
        state.append(0.0)
    return np.array(state, dtype=np.float32)

# ===============================
# Connect to CARLA
# ===============================
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(5.0)

try:
    world = client.get_world()
    print(f"Connected to CARLA on port {CARLA_PORT}")
except Exception as e:
    raise RuntimeError("Cannot connect to CARLA. Make sure server is running.") from e

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)

# ===============================
# Spawn ego vehicle
# ===============================
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_points = world.get_map().get_spawn_points()

vehicle = None
for sp in spawn_points:
    try:
        vehicle = world.spawn_actor(vehicle_bp, sp)
        print("Vehicle spawned at:", sp)
        break
    except RuntimeError:
        continue

if vehicle is None:
    raise RuntimeError("No free spawn points available")

# ===============================
# Attach camera to vehicle
# ===============================
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

spectator = world.get_spectator()

vehicle_transform = vehicle.get_transform()
spectator_transform = carla.Transform(
    vehicle_transform.location + carla.Location(x=-8, z=5),
    vehicle_transform.rotation
)
spectator.set_transform(spectator_transform)

# ===============================
# Setup Global Route Planner
# ===============================
planner = GlobalRoutePlanner(world.get_map(), sampling_resolution=SAMPLING_RESOLUTION)

start_location = spawn_points[0].location
end_location = spawn_points[-1].location

route = planner.trace_route(start_location, end_location)
waypoints = [wp for wp, _ in route]

# ===============================
# Main Loop: compute state vector
# ===============================
try:
    while True:
        world.tick()

        # Update ego
        ego_pos, ego_yaw = get_ego_transform(vehicle)

        # Update spectator to follow vehicle
        vehicle_transform = vehicle.get_transform()
        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(x=-8, z=5),
            vehicle_transform.rotation
        )
        spectator.set_transform(spectator_transform)

        # Closest waypoint
        distances = [np.linalg.norm(np.array([wp.transform.location.x, wp.transform.location.y]) - ego_pos[:2])
                     for wp in waypoints]
        closest_idx = int(np.argmin(distances))

        # State vector
        state_waypoints = waypoints[closest_idx:closest_idx + WAYPOINT_FRAME_SIZE]
        state_vector = build_state_vector(ego_pos, ego_yaw, state_waypoints)
        print("State vector:", state_vector)

        # Optional: autopilot
        vehicle.set_autopilot(True)

        time.sleep(0.05)


except KeyboardInterrupt:
    print("Exiting simulation.")

finally:
    # Cleanup
    camera.stop()
    camera.destroy()
    if vehicle is not None:
        vehicle.destroy()
    cv2.destroyAllWindows()
    print("Actors destroyed, simulation ended.")
