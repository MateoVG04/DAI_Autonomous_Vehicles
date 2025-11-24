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

# def build_state_vector(vehicle, waypoints, frame_size, lane_width, speed, accel, dist_to_obj_ahead):
#     """
#     Build a state vector for DRL input using only y-values of waypoints ahead.
#
#     Args:
#         vehicle: CARLA vehicle actor
#         waypoints: list of CARLA waypoints (global positions)
#         frame_size: int, number of future waypoints to include
#         lane_width: float, width of the lane for normalization
#         speed: float, current speed of the vehicle
#         accel: float, current acceleration of the vehicle
#         dist_to_obj_ahead: float, distance to car ahead
#
#     Returns:
#         np.ndarray of shape (frame_size,), x- and y-values
#     """
#     ego_pos, ego_yaw = get_ego_transform(vehicle)
#
#     max_distance = 40
#     max_offset = lane_width / 2
#     future_waypoints  = waypoints[:frame_size]
#     xy_local = []
#     for wp in future_waypoints:
#         local_coords = global_to_local(ego_pos, ego_yaw,
#             (wp.transform.location.x, wp.transform.location.y))
#         y_norm = np.clip(local_coords[1] / max_offset, -1, 1)
#         x_norm = np.clip(local_coords[0] / max_distance, 0, 1)
#         xy_local.extend([x_norm, y_norm])
#     while len(xy_local) < frame_size * 2:
#         xy_local.extend([0.0, 0.0])  # padding for missing waypoints
#
#     # Hardcoded for now (city traffic)
#     ref_speed = 15.0 # m/s (~54 km/h)
#     ref_accel = 10.0 # m/s²
#     speed_norm = np.clip(speed / ref_speed, 0, 1)
#     accel_norm = np.clip(accel / ref_accel, -1, 1)
#     dist_norm = np.clip(dist_to_obj_ahead / max_distance, 0, 1)
#
#     xy_local.extend([speed_norm, accel_norm, dist_norm])
#
#     return np.array(xy_local, dtype=np.float32)
#

def build_improved_state_vector(vehicle, waypoints, frame_size, current_steering, speed, dist_to_obj_ahead):
    """
    Args:
        vehicle: Carla actor
        waypoints: List of raw carla waypoints (dense list)
        frame_size: How many points to feed the network (e.g., 14)
        current_steering: The current wheel angle [-1, 1]
        speed: Current speed in m/s
        dist_to_obj_ahead: Distance in meters
    """
    ego_pos, ego_yaw = get_ego_transform(vehicle)

    # CONSTANTS
    MAX_DIST = 50.0  # Increased lookahead normalization range
    MAX_SPEED = 30.0  # 20.0 is low for highway/city combined

    # 1. DOWNSAMPLING (Crucial!)
    # Instead of taking indices 0,1,2... we take 0, 2, 4... or based on distance.
    # This ensures index 14 represents a point ~30+ meters away, not 7 meters.
    sample_resolution = 2  # Take every 2nd waypoint

    # Slice the dense list to get a sparse lookahead
    sparse_indices = [i * sample_resolution for i in range(frame_size)]
    future_waypoints = []
    for idx in sparse_indices:
        if idx < len(waypoints):
            future_waypoints.append(waypoints[idx])

    polar_coords = []

    for wp in future_waypoints:
        # Transform to Local Frame (Same as your logic)
        lx, ly = global_to_local(
            ego_pos, ego_yaw,
            (wp.transform.location.x, wp.transform.location.y)
        )

        # 2. CONVERT TO POLAR
        # Distance to waypoint
        dist = np.sqrt(lx ** 2 + ly ** 2)

        # Angle to waypoint (atan2 returns radians between -pi and pi)
        angle = np.arctan2(ly, lx)

        # Normalize
        dist_norm = np.clip(dist / MAX_DIST, 0, 1)

        # Normalize angle to [-1, 1] (assuming roughly +/- 90 degrees FOV)
        angle_norm = np.clip(angle / (np.pi / 2), -1, 1)

        polar_coords.extend([dist_norm, angle_norm])

    # Pad if we ran out of waypoints (e.g. end of track)
    expected_len = frame_size * 2
    while len(polar_coords) < expected_len:
        # Pad with "Max Distance" and "0 Angle" (Straight ahead far away)
        polar_coords.extend([1.0, 0.0])

    # 3. PHYSICS STATE
    speed_norm = np.clip(speed / MAX_SPEED, 0, 1)
    dist_obj_norm = np.clip(dist_to_obj_ahead / MAX_DIST, 0, 1)

    # Add CURRENT STEERING (Proprioception)
    # This helps the agent know "Rate of Change" needed
    steering_norm = np.clip(current_steering, -1, 1)

    # Calculate lateral error to the *immediate* closest path point (lane centering)
    # This acts as a "Cross Track Error" signal
    if len(future_waypoints) > 0:
        wp0 = future_waypoints[0]
        # A simple approximation of cross-track error is the local Y of the first point
        lx0, ly0 = global_to_local(ego_pos, ego_yaw, (wp0.transform.location.x, wp0.transform.location.y))
        cte_norm = np.clip(ly0 / 5.0, -1, 1)  # Normalize by lane width approx
    else:
        cte_norm = 0.0

    # Final Vector Assembly
    final_state = np.concatenate([
        np.array(polar_coords, dtype=np.float32),  # Path geometry
        np.array([speed_norm], dtype=np.float32),  # Dynamics
        np.array([steering_norm], dtype=np.float32),  # Proprioception
        np.array([cte_norm], dtype=np.float32),  # Immediate Error
        np.array([dist_obj_norm], dtype=np.float32)  # Safety
    ])

    return final_state