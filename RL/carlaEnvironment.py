import carla
import gymnasium as gym
import numpy as np
import Pyro4
import random

from agents.navigation.behavior_agent import BehaviorAgent
from RL.env_utils import build_state_vector

@Pyro4.expose
class CarlaEnv(gym.Env):
    def __init__(self, world, vehicle):
        super().__init__()

        # World and vehicle
        self.world = world
        self.vehicle = vehicle

        # Define action and observation space
        # negative action values correspond to braking, positive to throttle
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0, shape=(1,), dtype=np.float32)

        self.waypoints_size = 15 # Hardcoded for now
        self.lane_width = 3.5  # meters
        obs_size = self.waypoints_size + 3  # waypoints_size=15 + speed + accel + dist_to_car_ahead
        self.obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Behavior agent for navigation
        self.agent = BehaviorAgent(self.vehicle, behavior='normal')
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Radar setup
        self.max_dist_ahead = 40.0  # default max distance
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
        # Keep only objects roughly in front (±10° horizontal, ±2° vertical)
        forward_detections = [d for d in data if abs(np.degrees(d.azimuth)) < 10 and abs(np.degrees(d.altitude)) < 2]
        if forward_detections:
            # Find the closest object ahead
            self.max_dist_ahead = min(d.depth for d in forward_detections)
        else:
            self.max_dist_ahead = 40.0  # max distance


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
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        # Set the agent's destination
        self.agent.set_destination(destination.location)

        # Reset step counter
        self.episode_step = 0

        # Return initial state (toList for pyro serialization)
        return self.get_obs().tolist(), {}

    def step(self, action):
        # clip and apply control
        action = float(np.array(action).squeeze())
        action = np.clip(action, -1.0, 1.0)
        if action > 0:
            throttle = action
            brake = 0.0
        else:
            throttle = 0.0
            brake = -action
        agent_control = self.agent.run_step()

        # apply control of the RL agent for throttle and brake, and the behavior agent for steering
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=agent_control.steer)
        self.vehicle.apply_control(control)

        # Tick synchronous simulation
        self.world.tick()
        self.episode_step += 1

        # Get state, reward, done
        obs = self.get_obs()
        reward, terminated = self.compute_reward()

        truncated = False
        if self.episode_step > 1000:
            truncated = True

        # (toList for pyro serialization)
        return obs.tolist(), reward, terminated, truncated, {}

    def close(self):
        if self.radar_sensor:
            self.radar_sensor.stop()
            self.radar_sensor.destroy()
            self.radar_sensor = None

    def render(self):
        pass

    def get_obs(self):
        # Get plan
        agent_plan = self.agent.get_local_planner().get_plan()

        # Extract the carla.Waypoint objects (sorted)
        waypoints = [wp[0] for wp in agent_plan]

        # Get speed and acceleration
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        speed = np.linalg.norm([velocity.x, velocity.y])
        accel = np.linalg.norm([acceleration.x, acceleration.y])
        return build_state_vector(self.vehicle, waypoints, self.waypoints_size, self.lane_width, speed, accel, self.max_dist_ahead)

    # TODO: design a better reward function
    def compute_reward(self):
        """
        Simple reward function for RL driving:
        - Penalize small distance ahead
        - Penalize collision / bad braking
        - Reward reasonable speed and reaching destination
        """

        reward = 0.0
        terminated = False  # crash or done

        # get speed
        velocity = self.vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])

        target_speed = 15.0  # m/s (~54 km/h)

        speed_error = abs(speed - target_speed)
        reward -= 0.05 * (speed_error ** 2)  # quadratic penalty

        # Keep safe distance to car in front
        if self.max_dist_ahead < 5.0:  # very close to car ahead
            reward -= 10.0
            terminated = True
        elif self.max_dist_ahead < 15.0:  # getting too close
            reward -= 1.0

        # penalty for harsh braking
        brake = self.vehicle.get_control().brake
        if brake > 0.8:  # very harsh braking
            reward -= 0.5

        # Reward for reaching destination
        if self.agent.done():
            reward += 20.0  # Large reward for finishing
            terminated = True

        return reward, terminated


