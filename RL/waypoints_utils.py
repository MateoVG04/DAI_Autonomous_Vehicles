import math
import numpy as np
import carla



WAYPOINT_FRAME_SIZE = 15  # number of waypoints in the state vector
SAMPLING_RESOLUTION = 2.0  # meters between global waypoints


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
    x_local = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
    y_local = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
    return np.array([x_local, y_local])

def build_state_vector(ego_pos, ego_yaw, waypoints, frame_size, lane_width, speed, accel, dist):
    """
    Build a state vector for DRL input using only y-values of waypoints ahead.

    Args:
        ego_pos: np.array([x, y, z]) ego vehicle global position
        ego_yaw: float, ego vehicle yaw in radians
        waypoints: list of CARLA waypoints (global positions)
        frame_size: int, number of future waypoints to include
        lane_width: float, width of the lane for normalization
        speed: float, current speed of the vehicle
        accel: float, current acceleration of the vehicle
        dist: float, distance between waypoints

    Returns:
        np.ndarray of shape (frame_size,), only y-values
    """
    max_offset = lane_width / 2
    distances = [np.linalg.norm(np.array([wp.transform.location.x, wp.transform.location.y]) - ego_pos[:2])
                 for wp in waypoints]
    closest_idx = int(np.argmin(distances))
    future_waypoints = waypoints[closest_idx : closest_idx + frame_size]
    y_local_list = []
    for wp in future_waypoints:
        local_coords = global_to_local(ego_pos, ego_yaw,
                                       (wp.transform.location.x, wp.transform.location.y))
        y_norm = np.clip(local_coords[1] / max_offset, -1, 1)
        y_local_list.append(y_norm)
    while len(y_local_list) < frame_size:
        y_local_list.append(0.0)
    speed_norm = np.clip(speed / 30.0, 0, 1)
    accel_norm = np.clip(accel / 5.0, -1, 1)
    dist_norm = np.clip(dist / 50.0, 0, 1)
    y_local_list.extend([speed_norm, accel_norm, dist_norm])
    return np.array(y_local_list, dtype=np.float32)


def compute_safe_spawn_location_ahead(world, ego_vehicle, distance_ahead):
    ego_transform = ego_vehicle.get_transform()

    ego_waypoint = world.get_map().get_waypoint(ego_transform.location, project_to_road=True)

    # Get the waypoint that is distance_ahead in front
    next_waypoints = ego_waypoint.next(distance_ahead)
    if len(next_waypoints) == 0:
        raise RuntimeError("No waypoint found ahead at specified distance")

    spawn_waypoint = next_waypoints[0]  # first waypoint at that distance
    spawn_location = spawn_waypoint.transform.location
    spawn_rotation = spawn_waypoint.transform.rotation

    return carla.Transform(spawn_location, spawn_rotation)

def update_spectator(vehicle_transform, distance_behind, height_above):
    # Convert yaw (in degrees) to radians
    yaw = math.radians(vehicle_transform.rotation.yaw)

    # Compute forward vector of the vehicle
    forward_vector = carla.Vector3D(
        math.cos(yaw),
        math.sin(yaw),
        0
    )

    # Calculate the camera location relative to the vehicle
    camera_location = vehicle_transform.location - forward_vector * distance_behind
    camera_location.z += height_above

    spectator_transform = carla.Transform(
        camera_location,
        vehicle_transform.rotation
    )
    return spectator_transform


def get_distance_to_lead_vehicle(world, ego_vehicle, max_distance):
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()

    min_distance = max_distance
    lead_vehicle = None

    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego_vehicle.id:
            continue

        target_location = actor.get_transform().location
        direction = target_location - ego_location
        distance = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)

        # Only consider vehicles in front and within range
        forward_dot = ego_forward.x * direction.x + ego_forward.y * direction.y + ego_forward.z * direction.z
        if forward_dot > 0 and distance < min_distance:
            min_distance = distance
            lead_vehicle = actor

    if lead_vehicle:
        return min_distance
    else:
        return max_distance  # no vehicle ahead