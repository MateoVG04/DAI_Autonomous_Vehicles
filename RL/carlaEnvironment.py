import carla
import gymnasium as gym
from waypoints_utils import *
from agents.navigation.behavior_agent import BehaviorAgent

import random

class CarlaEnv(gym.Env):
    def __init__(self, world, vehicle):
        super().__init__()

        # World, vehicle and amount of waypoints to consider
        self.world = world
        self.vehicle = vehicle

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

        self.waypoints_size = 15 # Hardcoded for now
        obs_size = self.waypoints_size + 3  # waypoints_size=15 + speed + accel + dist_to_car_ahead
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_size,), dtype=np.float32)

        # Behavior agent for navigation
        self.agent = BehaviorAgent(self.vehicle, behavior='normal')
        self.spawn_points = self.world.get_map().get_spawn_points()

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

        # Episode step counter
        self.episode_step = 0

    def radar_callback(self, data):
        # Keep only objects roughly in front (±5° horizontal, ±2° vertical)
        forward_detections = [d for d in data if abs(math.degrees(d.azimuth)) < 5 and abs(math.degrees(d.altitude)) < 2]
        if forward_detections:
            self.max_dist_ahead = min(d.depth for d in forward_detections)
        else:
            self.max_dist_ahead = 50.0  # max distance

    def reset(self, seed=None, options=None):
        # Pick a random spawn point
        spawn_point = random.choice(self.spawn_points)

        # Pick a random destination that is NOT the spawn point
        destination = spawn_point
        while destination == spawn_point:
            destination = random.choice(self.spawn_points)

        # Set the vehicle to the spawn point
        self.vehicle.set_transform(spawn_point)

        # Reset vehicle physics
        self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_acceleration(carla.Vector3D(0, 0, 0))

        # Set the agent's destination
        self.agent.set_destination(destination.location)

        # Reset step counter
        self.episode_step = 0

        # Return initial state
        return self.get_state(), {}

    def step(self, action):
        # clip and apply control
        throttle = float(np.clip(action[0], 0.0, 1.0))
        brake = float(np.clip(action[1], 0.0, 1.0))
        agent_control = self.agent.run_step()

        # apply control of the RL agent for throttle and brake, and the behavior agent for steering
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=agent_control.steer)
        self.vehicle.apply_control(control)

        # Tick synchronous simulation
        self.world.tick()
        self.episode_step += 1

        # Get state, reward, done
        state = self.get_state()
        reward, terminated = self.compute_reward()

        truncated = False
        if self.episode_step > 1000:
            truncated = True

        info = {}

        return state, reward, terminated, truncated, info

    # TODO: Understand and implement waypoints and reward function
    # TODO: Train the agent
    def get_state(self):
        # This returns a list of (carla.Waypoint, RoadOption)
        agent_plan = self.agent.get_local_planner().get_plan()

        # Now, you need to extract just the carla.Waypoint objects
        waypoints = [wp[0] for wp in agent_plan]

        ego_pos, ego_yaw = get_ego_transform(self.vehicle)
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        accel = np.linalg.norm([self.vehicle.get_acceleration().x,
                                self.vehicle.get_acceleration().y])
        dist_to_car_ahead = self.max_dist_ahead
        return build_state_vector(ego_pos, ego_yaw, waypoints, self.waypoints_size, 3.5, speed, accel, dist_to_car_ahead)

    def compute_reward(self):
        reward = 0.0
        terminated = False  # crash or done

        velocity = self.vehicle.get_velocity()
        speed_kmh = np.linalg.norm([velocity.x, velocity.y])

        target_speed = 30

        # 1. Reward for being at the target speed (e.g., within a margin)
        speed_error = abs(speed_kmh - target_speed)
        # Use a non-linear penalty to punish large errors more
        reward -= (speed_error ** 2) * 0.01

        # 2. Reward for safe following distance (using your radar)
        if self.max_dist_ahead < 30.0:
            # Reward for being close, but penalize for being TOO close
            # This creates a "sweet spot"
            safe_distance_reward = 1.0 - (abs(self.max_dist_ahead - 15.0) / 15.0)  # Peaks at 15m
            reward += safe_distance_reward * 0.05

        # 3. Penalize heavily for near-crash
        if self.max_dist_ahead < 5.0:
            reward -= 10.0
            terminated = True  # This is a terminal failure state

        # 4. Reward for reaching destination
        if self.agent.done():
            reward += 20.0  # Large reward for finishing
            terminated = True

        return reward, terminated
