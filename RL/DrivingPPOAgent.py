import csv
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from shapely.geometry import Polygon, MultiPolygon
import carla
from RL.ACNetwork import ActorCriticNetwork
from RL.carla_environment import CarlaEnv
from agents.tools.misc import compute_distance, get_speed

import math

def save_metrics_to_csv(filename, metrics_dict):
    """Save metrics to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write metrics to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['iteration', 'loss', 'average_reward'])
        # Write data
        for i in range(len(metrics_dict['loss'])):
            writer.writerow([i,
                             metrics_dict['loss'][i],
                             metrics_dict['reward'][i]])

class DrivingPPOAgent:
    def __init__(self, env, learning_rate=3e-4, clip_range=0.2,
                 value_loss_coef=0.5, max_grad_norm=0.5, vehicle=None):
        self.device = torch.device("cpu")

        # TODO Still need to get the dimensions. These might change depending on what the computer vision does
        self.in_steer_dim = ...
        self.out_steer_dim = ...
        self.in_speed_dim = ...
        self.out_speed_dim = ...

        self.vehicle

        self.network = ActorCriticNetwork(input_steering_dim=self.in_steer_dim, output_steering_dim=self.out_steer_dim, input_speed_dim=self.in_speed_dim, output_speed_dim=self.out_speed_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, images, current_speed, distances_to_vehicle=None, speed_signs=None, pedestrians=None, red_lights=None):
        with torch.no_grad():
            action, log_prob = self.network.sample_action(images, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights)
        return action.cpu().numpy()[0], log_prob.cpu().item()

    def update(self, all_episodes_data, epochs:int=10):
        """
        Update policy using data from multiple episodes

        Args:
            all_episodes_data: list of dictionaries, each containing:
                - states: numpy array of states
                - actions: numpy array of actions
                - advantages: numpy array of advantages
                - old_log_probs: numpy array of old log probabilities
                - returns: numpy array of returns
        """
        all_states = np.concatenate([ep['states'] for ep in all_episodes_data])
        all_actions = np.concatenate([ep['actions'] for ep in all_episodes_data])
        all_advantages = np.concatenate([ep['advantages'] for ep in all_episodes_data])
        all_old_log_probs = np.concatenate([ep['log_probs'] for ep in all_episodes_data])
        all_returns = np.concatenate([ep['returns'] for ep in all_episodes_data])

        images, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights = torch.FloatTensor(all_states).to(self.device)
        actions = torch.FloatTensor(all_actions).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0
        for _ in range(epochs):
            action_mean, action_std, state_values = self.network(images, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights)
            dist = torch.distributions.Normal(action_mean, action_std)

            raw_actions = torch.atanh(actions.clamp(-0.99, 0.99))
            log_probs = dist.log_prob(raw_actions)
            log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - state_values.squeeze()).pow(2).mean()

            # For exploration
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_loss_coef * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / epochs

    def reward_function(self, images, current_speed, distances_to_vehicle, speed_signs, pedestrians, red_lights):
        """
        The reward function tells the agent if it doing a good action, or punishes
        :param images:
        :param current_speed:
        :param distances_to_vehicle:
        :param speed_signs:
        :param pedestrians:
        :param red_lights:
        :return:
        """
        # TODO Figuring out how we get into the environment and how we can see the next state!
        ...

def setup_player(world):
    # Get a random blueprint.
    blueprint_library = world.get_blueprint_library()
    blueprint = random.choice(blueprint_library.filter('vehicle.*'))
    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    # Spawn point selection
    spawn_attempts = 0
    actor = random_spawn(world, blueprint)
    while actor is None and spawn_attempts < 20:
        actor = random_spawn(world, blueprint)
        spawn_attempts += 1
    if actor is None:
        print("Could not spawn actor in 20 attempts")
        raise

    physics_control = actor.get_physics_control()
    physics_control.use_sweep_wheel_collision = True
    actor.apply_physics_control(physics_control)
    return actor

def random_spawn(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
    spawn_point.location.z += 2.0
    spawn_point.rotation.roll = 0.0
    spawn_point.rotation.pitch = 0.0
    return world.try_spawn_actor(blueprint, spawn_point)


def run_episode(env, agent, vehicle, images, distance_to_vehicle, speed_signs=None, pedestrians=None, red_lights=None):
    """
    Run a single episode and return the collected data
    The images are what the camera captures or the processed image for lane detection (given by the
    computer vision module)
    distance_to_vehicle: given by computer vision module
    speed_signs: given by computer vision module or by carla
    pedestrians: given by computer vision module
    red_lights: given by computer vision module
    """
    state, _ = env.reset()
    current_speed = get_speed(vehicle)
    if speed_signs is None:
        speed_signs = vehicle.get_speed_signs()

    done = False
    states, actions, rewards, values, log_probs = [], [], [], [], []
    episode_reward = 0

    while not done:
        states.append(state)
        action, log_prob = agent.select_action(images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights)

        with torch.no_grad():
            _, _, value = agent.network(images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights)

        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.item())

        # TODO fix a environment where we can do steps -> world ticks
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        episode_reward += reward

        done = terminated or truncated
        state = next_state

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    values = np.array(values)
    log_probs = np.array(log_probs)

    # Compute advantages and returns
    with torch.no_grad():
        _, _, next_value = agent.network(images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights)

    advantages = compute_gae(rewards, values, np.array([False] * len(rewards) + [done]),
                             next_value.item())
    returns = advantages + values

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'advantages': advantages,
        'log_probs': log_probs,
        'returns': returns,
        'episode_reward': episode_reward
    }


def main():
    num_iterations = 500  # Number of policy updates
    num_episodes_per_update = 5  # Number of episodes to collect before updating
    eval_frequency = 100

    # Environment setup
    car_pos = [2060.0, -50.0, 2]
    target_pos = [11530.0, 13000.0, 0]

    client = carla.Client("localhost", 2000)
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    print(client.get_available_maps())

    client.set_timeout(10.0)
    env = client.get_world()
    vehicle = setup_player(world=env)
    world = CarlaEnv(client=client)

    # Initialize agent
    agent = DrivingPPOAgent(env, learning_rate=3e-4, vehicle=vehicle)

    # Training metrics
    metrics = {
        'loss': [],
        'reward': []
    }

    print("Starting training...")

    for iteration in range(num_iterations):
        episode_data = []
        total_reward = 0
        #TODO fix speed_sign
        speed_signs = ...
        images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights = world.get_state_info(player=vehicle,speed_signs=speed_signs)
        # Collect episodes with different initial conditions
        for episode in range(num_episodes_per_update):
            # Run episode
            episode_result = run_episode(env, agent, vehicle, images, distance_to_vehicle, speed_signs, pedestrians, red_lights)
            episode_data.append(episode_result)
            total_reward += episode_result['episode_reward']

        # Update policy using all collected episodes
        loss = agent.update(episode_data)

        # Store metrics
        avg_reward = total_reward / num_episodes_per_update
        metrics['loss'].append(loss)
        metrics['reward'].append(avg_reward)

        if iteration % eval_frequency == 0:
            print(f"Iteration {iteration}, Average Reward: {avg_reward:.3f}, Loss: {loss:.3f}")

            # Save checkpoint
            if iteration > 0 and iteration % 100 == 0:
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.network.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, f'results/ship_ppo_checkpoint_{iteration}.pt')

    print("\nTraining completed!")
    env.close()


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    rewards: array of rewards for the batch
    values: array of value estimates
    dones: array of done flags
    next_value: value estimate for the state after the batch
    gamma: discount factor
    lambda_: GAE parameter
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0

    # Reverse iteration for GAE calculation
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae

    return advantages


def evaluate_trained_agent(checkpoint_path, num_episodes=3):
    """
    Evaluate the trained agent.
    """
    # Set up environment without rendering
    car_pos = [2060.0, -50.0]
    target_pos = [-1530.0, 12010.0]
    client = carla.Client("localhost", 2000)
    server_version = client.get_server_version()
    client_version = client.get_client_version()

    client.set_timeout(10.0)
    world = CarlaEnv(client=client)

    logger, end_simulation, env, tracer, agent, distance_hist, speed_hist, loop_count, player = world.do_carla_setup()

    agent = DrivingPPOAgent(env)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agent.network.load_state_dict(checkpoint['model_state_dict'])
    agent.network.eval()

    print(f"\nRunning {num_episodes} episodes...")

    all_paths = []
    total_rewards = []
    all_cross_errors = []  # Store cross-track errors for each episode

    for episode in range(num_episodes):
        state, _ = env.reset()
        # TODO fix the input data (the state)
        speed_signs = ...
        images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights = world.get_state_info(player, speed_signs)

        world.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = agent.select_action(images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights)

            next_state, reward, terminated, truncated, _ = world.step(action)
            done = terminated or truncated
            images, current_speed, distance_to_vehicle, speed_signs, pedestrians, red_lights = next_state
            episode_reward += reward
            steps += 1

            if done:
                print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                break
        total_rewards.append(episode_reward)

    # Print summary statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nEvaluation completed!")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Best episode reward: {max(total_rewards):.2f}")
    print(f"Worst episode reward: {min(total_rewards):.2f}")

    #######################
    # TODO Make/Show here the plots of your evaluation
    #######################

    env.close()
    return all_paths, total_rewards, all_cross_errors


if __name__ == "__main__":
    # Uncomment for evaluation
    checkpoint_path = "DrivingPPOAgent.pt"  # Adjust to your checkpoint file
    evaluate_trained_agent(checkpoint_path)
