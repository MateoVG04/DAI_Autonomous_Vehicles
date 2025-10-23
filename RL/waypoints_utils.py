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
    x_local = math.cos(-ego_yaw) * dx - math.sin(-ego_yaw) * dy
    y_local = math.sin(-ego_yaw) * dx + math.cos(-ego_yaw) * dy
    return np.array([x_local, y_local])

def build_state_vector(ego_pos, ego_yaw, waypoints):
    state = []
    for wp in waypoints[:WAYPOINT_FRAME_SIZE]:
        local_coords = global_to_local(
            ego_pos, ego_yaw,
            (wp.transform.location.x, wp.transform.location.y)
        )
        state.extend(local_coords)  # (x, y)
    while len(state) < 2 * WAYPOINT_FRAME_SIZE:
        state.extend([0.0, 0.0])
    return np.array(state, dtype=np.float32)

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

import carla
import math
import numpy as np

class LeadVehicleRadar:
    def __init__(self, world, ego_vehicle, forward_distance=50.0, fov_horizontal=30, fov_vertical=5, z_offset=1.0, x_offset=2.5):
        """
        world: CARLA world object
        ego_vehicle: your ego vehicle actor
        forward_distance: max radar range (meters)
        fov_horizontal: horizontal field of view (degrees)
        fov_vertical: vertical field of view (degrees)
        z_offset: height above ego vehicle for sensor
        x_offset: forward offset from ego vehicle for sensor
        """
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.forward_distance = forward_distance
        self.latest_distance = forward_distance

        # Create radar blueprint
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(fov_horizontal))
        bp.set_attribute('vertical_fov', str(fov_vertical))
        bp.set_attribute('range', str(forward_distance))

        # Attach radar in front of vehicle
        transform = carla.Transform(carla.Location(x=x_offset, z=z_offset))
        self.sensor = world.spawn_actor(bp, transform, attach_to=ego_vehicle)

        # Register callback
        self.sensor.listen(self._radar_callback)

    def _radar_callback(self, radar_data):
        """Process radar detections and find closest object ahead"""
        # Filter points roughly in front
        forward_detections = [
            d for d in radar_data
            if abs(math.degrees(d.azimuth)) < 5 and abs(math.degrees(d.altitude)) < 2
        ]
        if forward_detections:
            distances = [d.depth for d in forward_detections]
            self.latest_distance = min(distances)
        else:
            self.latest_distance = self.forward_distance

    def get_distance(self):
        """Returns the latest distance to lead vehicle (meters)"""
        return self.latest_distance

    def destroy(self):
        """Cleanup the sensor"""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
