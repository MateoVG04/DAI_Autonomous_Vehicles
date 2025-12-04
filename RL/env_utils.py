import math
import carla
import numpy as np

"""Helper functions for building the state vector (observation) for the DRL agent."""

def get_vehicle_speed_accel(vehicle: carla.Vehicle):
    """Returns vehicle speed (m/s) and acceleration (m/s²)."""
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    accel = vehicle.get_acceleration()
    acceleration = math.sqrt(accel.x**2 + accel.y**2 + accel.z**2)

    return speed, acceleration

def get_ego_transform(vehicle: carla.Vehicle):
    """Returns ego vehicle position and yaw angle in radians."""
    transform = vehicle.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    return np.array([loc.x, loc.y, loc.z]), yaw

def global_to_local(ego_pos: np.array, ego_yaw: float, waypoint_pos: tuple):
    """Transforms a waypoint from global to ego-local coordinates."""
    dx = waypoint_pos[0] - ego_pos[0]
    dy = waypoint_pos[1] - ego_pos[1]
    x_local = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
    y_local = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
    return np.array([x_local, y_local])

def build_state_vector(vehicle, waypoints, frame_size, lane_width, speed, accel, steering, dist_to_obj_ahead):
    """
    Builds the state vector (observation) for the DRL agent.
    - Waypoints are converted to Polar Coordinates (Distance, Angle) in ego-local frame.
    - Includes proprioceptive features: speed, acceleration, steering command.
    - Includes safety feature: distance to car ahead.
    - Normalizations applied to keep values in [-1, 1].

    :arg:
        vehicle: Carla.Vehicle, CARLA vehicle actor
        waypoints: list, list of CARLA waypoints
        frame_size: int, number of future waypoints to include
        lane_width: float, width of the lane for normalization
        speed: float, current speed of the vehicle (m/s)
        accel: float, current acceleration of the vehicle (m/s²)
        steering: float, current steering control [-1, 1]
        dist_to_obj_ahead: float, distance to object ahead (m)

    :return:
        np.ndarray of shape (2 * frame_size + 5,), the normalized state vector.
    """

    # Get ego vehicle position and orientation
    ego_pos, ego_yaw = get_ego_transform(vehicle)

    # --- CONSTANTS ---
    MAX_LOOKAHEAD_DIST = 50.0  # Max physical range for waypoints (m)
    MAX_REF_LATERAL = lane_width / 2.0  # Max lateral offset for normalization (m)
    REF_SPEED = 15.0  # Reference speed for normalization (~54 km/h)
    REF_ACCEL = 10.0  # Reference acceleration (m/s²)

    # --- 1. Waypoint Filtering & Transformation (Polar) ---

    polar_coords = []
    local_coords = []

    # Convert waypoints to ego-local coordinates
    for wp in waypoints:
        coords = global_to_local(
            ego_pos, ego_yaw, (wp.transform.location.x, wp.transform.location.y)
        )
        local_coords.append(coords)

    # Convert local coordinates to Polar Coordinates and normalize
    for lx, ly in local_coords:

        # Distance (r)
        dist = np.sqrt(lx ** 2 + ly ** 2)

        # Angle (theta) - The primary steering error signal
        angle_rad = np.arctan2(ly, lx)

        # Normalize Angle by the maximum useful angle (+/- 90 degrees or pi/2)
        angle_norm = np.clip(angle_rad / (np.pi / 2), -1, 1)

        # Normalize Distance by max lookahead range
        dist_norm = np.clip(dist / MAX_LOOKAHEAD_DIST, 0, 1)

        # Append to state vector
        polar_coords.extend([dist_norm, angle_norm])

    # Padding: Fill the rest of the vector with the "Far, Straight Ahead" signal
    while len(polar_coords) < frame_size * 2:
        polar_coords.extend([1.0, 0.0])

    # --- 2. Proprioceptive & Safety Features ---

    # Dynamics
    speed_norm = np.clip(speed / REF_SPEED, 0, 1)
    accel_norm = np.clip(accel / REF_ACCEL, -1, 1)

    # Proprioception: Get the current wheel command being applied by the agent
    steering_norm = np.clip(steering, -1, 1)

    # Lane Centering (Cross-Track Error - CTE)
    if len(local_coords) > 0:
        cte_norm = np.clip(local_coords[0][1] / MAX_REF_LATERAL, -1, 1)
    else:
        cte_norm = 0.0

    # Safety
    dist_norm = np.clip(dist_to_obj_ahead / MAX_LOOKAHEAD_DIST, 0, 1)

    # --- 3. Final Vector Assembly ---
    final_state = np.concatenate([
        np.array(polar_coords, dtype=np.float32),  # Path Geometry (Angle/Distance)
        np.array([speed_norm], dtype=np.float32),  # Dynamics
        np.array([accel_norm], dtype=np.float32),  # Acceleration
        np.array([steering_norm], dtype=np.float32),  # Current Control Command (Proprioception)
        np.array([cte_norm], dtype=np.float32),  # Immediate Lateral Error (Lane Centering)
        np.array([dist_norm], dtype=np.float32)  # Radar Distance
    ])

    return final_state