import math
import numpy as np
import carla

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

def build_state_vector(vehicle, waypoints, frame_size, lane_width, speed, accel, dist_to_car_ahead):
    """
    Build a state vector for DRL input using only y-values of waypoints ahead.

    Args:
        vehicle: CARLA vehicle actor
        waypoints: list of CARLA waypoints (global positions)
        frame_size: int, number of future waypoints to include
        lane_width: float, width of the lane for normalization
        speed: float, current speed of the vehicle
        accel: float, current acceleration of the vehicle
        dist_to_car_ahead: float, distance to car ahead

    Returns:
        np.ndarray of shape (frame_size,), x- and y-values
    """
    ego_pos, ego_yaw = get_ego_transform(vehicle)

    max_distance = 50
    max_offset = lane_width / 2
    future_waypoints  = waypoints[:frame_size]
    xy_local = []
    for wp in future_waypoints:
        local_coords = global_to_local(ego_pos, ego_yaw,
                                       (wp.transform.location.x, wp.transform.location.y))
        y_norm = np.clip(local_coords[1] / max_offset, -1, 1)
        x_norm = np.clip(local_coords[0] / max_distance, 0, 1)
        xy_local.extend([x_norm, y_norm])
    while len(xy_local) < frame_size * 2:
        xy_local.extend([0.0, 0.0])  # padding for missing waypoints

    # Hardcoded for now (city traffic)
    ref_speed = 15.0 # m/s (~54 km/h)
    ref_accel = 10.0 # m/s²
    speed_norm = np.clip(speed / ref_speed, 0, 1)
    accel_norm = np.clip(accel / ref_accel, -1, 1)
    dist_norm = np.clip(dist_to_car_ahead / max_distance, 0, 1)

    xy_local.extend([speed_norm, accel_norm, dist_norm])
    return np.array(xy_local, dtype=np.float32)

