import gymnasium as gym
import numpy as np
import Pyro4
import random

from agents.tools.misc import *
from RL.env_utils import build_state_vector
from RL.steering_agent import SteeringAgent

@Pyro4.expose
class CarlaEnv(gym.Env):
    def __init__(self, world, vehicle):
        super().__init__()

        # World and vehicle
        self.world = world
        self.ego_vehicle = vehicle

        # Define action and observation space
        # negative action values correspond to braking, positive to throttle
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0, shape=(1,), dtype=np.float32)

        self.waypoints_size = 15 # Hardcoded for now
        self.lane_width = 3.5  # meters
        obs_size = self.waypoints_size + 3  # waypoints_size=15 + speed + accel + dist_to_car_ahead
        self.obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Behavior agent for navigation
        self.agent = SteeringAgent(self.ego_vehicle, behavior='normal')
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Radar setup
        self.max_dist_ahead = 50.0  # default max distance
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '5')
        radar_bp.set_attribute('range', '40')
        radar_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        self.radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=self.ego_vehicle)
        self.radar_sensor.listen(self.radar_callback)

        # Episode step counter
        self.episode_step = 0

    def radar_callback(self, data):
        # Keep only objects roughly in front (±10° horizontal, ±2° vertical)
        forward_detections = [d for d in data]
        if forward_detections:
            # Find the closest object ahead
            self.max_dist_ahead = min(d.depth for d in forward_detections)
        else:
            self.max_dist_ahead = 50.0  # max distance

    def reset(self, seed=None, options=None):
        # Pick a random spawn point
        spawn_point = random.choice(self.spawn_points)

        # Pick a random destination that is NOT the spawn point
        # Create a list excluding the spawn point
        possible_destinations = [p for p in self.spawn_points if p != spawn_point]

        # Pick a random destination from the filtered list
        destination = random.choice(possible_destinations)

        # Set the vehicle to the spawn point
        self.ego_vehicle.set_transform(spawn_point)

        # Reset vehicle physics
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

        # Set the agent's destination
        self.agent.set_destination(destination.location)

        self.max_dist_ahead = 50.0

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
        self.ego_vehicle.apply_control(control)

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

        waypoints = self.agent.getWaypoints()

        # Get speed and acceleration
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        speed = np.linalg.norm([velocity.x, velocity.y])
        accel = np.linalg.norm([acceleration.x, acceleration.y])
        return build_state_vector(self.ego_vehicle, waypoints, self.waypoints_size, self.lane_width, speed, accel, self.max_dist_ahead)

    # TODO: design a better reward function
    def compute_reward(self):
        reward = 0.0
        terminated = False

        # --- Goal reached ---
        if self.agent.done():
            reward += 100.0
            terminated = True

        # --- Speed reward ---
        target_speed = 15.0
        speed = get_speed(self.ego_vehicle) / 3.6  # km/h → m/s
        reward += -0.05 * ((speed - target_speed) / target_speed) ** 2  # normalized

        # --- Vehicle ahead ---
        waypoints = self.agent.getWaypoints()
        vehicle_state, vehicle, distance = self.agent.collision_and_car_avoid_manager(waypoint=waypoints[0])
        if vehicle_state and vehicle is not None:
            safe_distance = max(5.0, min(15.0, distance))
            # Proportional penalty
            reward += -2.0 * (1.0 - safe_distance / 15.0)
            if distance < 5.0:
                terminated = True
            else:
                # match speed of leading vehicle
                lead_speed = get_speed(vehicle) / 3.6
                reward += -0.1 * abs(speed - lead_speed)

        # --- Tailgating ---
        if self.agent.behavior.tailgate_counter > 0:
            reward -= 2.0

        # --- Pedestrian collision ---
        walker_state, walker, w_distance = self.agent.pedestrian_avoid_manager(waypoint=waypoints[0])
        if walker_state and w_distance < 5.0:
            reward -= 10.0
            terminated = True
        elif walker_state:
            reward -= 1.0 * (1.0 - w_distance / 10.0)  # small continuous penalty

        # --- Smoothness reward ---
        control = self.ego_vehicle.get_control()
        reward -= 0.5 * control.brake ** 2
        reward -= 0.2 * control.throttle ** 2

        # --- Progress reward ---
        reward += 0.05 * speed

        return reward, terminated




