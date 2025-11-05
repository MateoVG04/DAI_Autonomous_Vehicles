import gymnasium as gym
from waypoints_utils import *


class CarlaEnv(gym.Env):
    def __init__(self, world, vehicle, waypoints, frame_size):
        super().__init__()
        self.world = world
        self.vehicle = vehicle
        self.waypoints = waypoints
        self.frame_size = frame_size

        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)
        obs_size = frame_size + 3
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_size,), dtype=np.float32)

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Radar setup
        self.max_dist_ahead = 50.0
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '5')
        radar_bp.set_attribute('range', '50')


        radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        self.radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        self.radar_sensor.listen(self.radar_callback)

        self.episode_step = 0

    def radar_callback(self, data):
        # Keep only objects roughly in front (±5° horizontal, ±2° vertical)
        forward_detections = [d for d in data if abs(math.degrees(d.azimuth)) < 5 and abs(math.degrees(d.altitude)) < 2]
        if forward_detections:
            self.max_dist_ahead = min(d.depth for d in forward_detections)
        else:
            self.max_dist_ahead = 50.0  # max distance

    def reset(self):
        # Reset vehicle
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle.set_transform(spawn_point)
        self.vehicle.set_velocity(carla.Vector3D(0,0,0))
        self.vehicle.set_angular_velocity(carla.Vector3D(0,0,0))
        self.vehicle.set_acceleration(carla.Vector3D(0,0,0))
        self.episode_step = 0
        return self.get_state()

    def step(self, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0)
        self.vehicle.apply_control(control)

        # Tick synchronous simulation
        self.world.tick()
        self.episode_step += 1

        state = self.get_state()
        reward, done = self.compute_reward()
        return state, reward, done, {}

    def get_state(self):
        ego_pos, ego_yaw = get_ego_transform(self.vehicle)
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        accel = np.linalg.norm([self.vehicle.get_acceleration().x,
                                self.vehicle.get_acceleration().y])
        dist_to_car_ahead = self.max_dist_ahead
        return build_state_vector(ego_pos, ego_yaw, self.waypoints, self.frame_size, 3.5, speed, accel, dist_to_car_ahead)

    def compute_reward(self):
        """
        Compute reward based on current state of the environment.
        :return:
        reward: float
        done: bool
        """
        reward = 0.0
        done = False

        # Basic speed-based reward
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        reward += speed * 0.05  # scaled speed reward

        # Penalize if car too close
        if self.max_dist_ahead < 5.0:
            reward -= 10.0
            done = True


        if self.episode_step > 1000:
            done = True

        return reward, done