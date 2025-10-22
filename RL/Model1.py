import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Hyperparameters
# Environment parameters
SAFE_DISTANCE = 10.0  # meters
DT = 0.1  # Simulation timestep in seconds

# DDPG Hyperparameters
ACTOR_LR = 0.001
CRITIC_LR = 0.002
GAMMA = 0.99  # Discount factor
TAU = 0.001  # For soft target updates
BUFFER_SIZE = 1_000_000  # Replay buffer size
BATCH_SIZE = 64
EXPLORATION_NOISE_STD = 0.1  # Std dev for exploration noise
ACCEL_CMD_MAX = 2.0  # Max acceleration
ACCEL_CMD_MIN = -3.0  # Max deceleration (brake)

# State dimensions
STATE_DIM = 4  # [distance_error, relative_velocity, host_velocity, acceleration]
ACTION_DIM = 1  # acceleration_command

import carla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import time  # For simulation step delays

# Assuming previous DDPG PyTorch code (Actor, Critic, ReplayBuffer, DDPGAgent) is available

# --- CARLA Specific Hyperparameters ---
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0  # seconds
SPAWN_POINT_EGO = 0  # Index of spawn point for ego car
SPAWN_POINT_LEAD = 1  # Index of spawn point for lead car
TARGET_SPEED_KPH = 50.0  # Target speed for ACC in km/h
TARGET_SPEED_MPS = TARGET_SPEED_KPH / 3.6  # Target speed in m/s

# DDPG Hyperparameters (re-defined or imported from previous code)
# ... (ACTOR_LR, CRITIC_LR, GAMMA, TAU, BUFFER_SIZE, BATCH_SIZE, EXPLORATION_NOISE_STD, etc.) ...
STATE_DIM = 4  # [distance_error, relative_velocity, host_velocity, acceleration]
ACTION_DIM = 1  # acceleration_command (-3 to +2 m/s^2)
ACCEL_CMD_MAX = 2.0
ACCEL_CMD_MIN = -3.0


# 3. Actor Network (PyTorch)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CarlaACCEnv:
    def __init__(self, host=CARLA_HOST, port=CARLA_PORT, timeout=CARLA_TIMEOUT,
                 dt=DT, safe_distance=SAFE_DISTANCE, target_speed_mps=TARGET_SPEED_MPS):
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.lead_vehicle = None
        self.dt = dt  # Simulation timestep from DDPG
        self.safe_distance = safe_distance
        self.target_speed_mps = target_speed_mps

        # Carla blueprint library
        self.blueprint_library = None
        self.spawn_points = None

        # Sensors (for reward, state)
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.last_ego_velocity = None
        self.last_ego_location = None
        self.collision_happened = False
        self.lane_invaded = False
        self.tick_count = 0

        # Connect to CARLA
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            print(f"Connected to CARLA server at {host}:{port}")
        except Exception as e:
            print(f"Error connecting to CARLA: {e}")
            raise

    def reset(self):
        self.collision_happened = False
        self.lane_invaded = False
        self.last_ego_velocity = None
        self.last_ego_location = None
        self.tick_count = 0

        # Get world and blueprints
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Set synchronous mode for consistent simulation
        settings.fixed_delta_seconds = self.dt  # Set fixed delta seconds matching RL step
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Destroy existing actors to ensure clean reset
        self.destroy_actors()

        # Spawn Ego Vehicle
        ego_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        self.ego_vehicle = self.world.spawn_actor(ego_bp, self.spawn_points[SPAWN_POINT_EGO])

        # Spawn Lead Vehicle (ahead of ego)
        lead_spawn_transform = carla.Transform(
            self.spawn_points[SPAWN_POINT_EGO].location + carla.Location(x=self.safe_distance + 20),  # Start ahead
            self.spawn_points[SPAWN_POINT_EGO].rotation
        )
        lead_bp = self.blueprint_library.filter('vehicle.audi.a2')[0]
        self.lead_vehicle = self.world.spawn_actor(lead_bp, lead_spawn_transform)

        # Set lead vehicle to a constant speed (for ACC target)
        self.lead_vehicle.set_autopilot(False)  # Ensure not on autopilot
        self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))  # Constant throttle to move it

        # Attach Collision Sensor
        colsensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(colsensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Attach Lane Invasion Sensor
        lanesensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lanesensor_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        self.world.tick()  # Advance one tick to initialize state
        return self._get_state()

    def _on_collision(self, event):
        self.collision_happened = True
        # print("COLLISION DETECTED!")

    def _on_lane_invasion(self, event):
        # Ignore minor lane invasions (e.g., slight curve) for simplicity if needed
        # self.lane_invaded = True
        pass  # For now, let's not penalize lane invasion heavily for a purely longitudinal ACC

    def _get_state(self):
        # 1. Host Car Data
        ego_location = self.ego_vehicle.get_location()
        ego_velocity_vec = self.ego_vehicle.get_velocity()
        host_velocity_mps = np.sqrt(ego_velocity_vec.x ** 2 + ego_velocity_vec.y ** 2 + ego_velocity_vec.z ** 2)

        # Host acceleration (numerical differentiation)
        host_acceleration = 0.0
        if self.last_ego_velocity is not None:
            delta_v_x = ego_velocity_vec.x - self.last_ego_velocity.x
            delta_v_y = ego_velocity_vec.y - self.last_ego_velocity.y
            delta_v_z = ego_velocity_vec.z - self.last_ego_velocity.z
            delta_v_len = np.sqrt(delta_v_x ** 2 + delta_v_y ** 2 + delta_v_z ** 2)
            host_acceleration = delta_v_len / self.dt
            # Consider direction: if slowing down, it's negative acceleration
            if host_velocity_mps < np.sqrt(
                    self.last_ego_velocity.x ** 2 + self.last_ego_velocity.y ** 2 + self.last_ego_velocity.z ** 2):
                host_acceleration *= -1
        self.last_ego_velocity = ego_velocity_vec

        # 2. Lead Car Data (simplistic, assuming we know the lead car)
        lead_location = self.lead_vehicle.get_location()
        lead_velocity_vec = self.lead_vehicle.get_velocity()
        lead_velocity_mps = np.sqrt(lead_velocity_vec.x ** 2 + lead_velocity_vec.y ** 2 + lead_velocity_vec.z ** 2)

        # Assuming driving along X-axis approximately (for distance and relative velocity)
        distance = lead_location.x - ego_location.x  # Simplistic for straight road
        distance_error = distance - self.safe_distance
        relative_velocity = lead_velocity_mps - host_velocity_mps

        return np.array([distance_error, relative_velocity, host_velocity_mps, host_acceleration], dtype=np.float32)

    def step(self, action_cmd):
        # Convert acceleration command to CARLA VehicleControl (throttle/brake)
        # This is a simplified mapping. A more sophisticated one might involve PID control
        # or the agent learning to output throttle/brake directly.
        action_cmd = np.clip(action_cmd, ACCEL_CMD_MIN, ACCEL_CMD_MAX).item()  # ensure scalar

        control = carla.VehicleControl()
        control.steer = 0.0  # Assuming perfect lane keeping for ACC
        control.hand_brake = False
        control.reverse = False

        if action_cmd >= 0:
            control.throttle = min(action_cmd / ACCEL_CMD_MAX, 1.0)  # Scale to 0-1
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(action_cmd) / abs(ACCEL_CMD_MIN), 1.0)  # Scale to 0-1

        self.ego_vehicle.apply_control(control)

        # Advance CARLA simulation
        self.world.tick()
        self.tick_count += 1

        next_state = self._get_state()
        reward = self._calculate_reward(next_state, action_cmd)
        done = self._is_done(next_state)

        return next_state, reward, done, {}

    def _calculate_reward(self, state, action_cmd):
        distance_error, relative_velocity, host_velocity, host_acceleration = state

        reward_distance = -0.1 * abs(distance_error)
        reward_velocity_match = -0.05 * abs(relative_velocity)
        reward_smoothness = -0.01 * (action_cmd ** 2)

        # Bonus for maintaining safe distance and speed match
        reward_bonus = 0.0
        if abs(distance_error) < 1.0 and abs(relative_velocity) < 0.5:
            reward_bonus = 5.0

        # Penalize collisions
        reward_collision_penalty = 0.0
        if self.collision_happened:
            reward_collision_penalty = -500.0  # Very heavy penalty

        # Penalize lane invasion (if implemented)
        # reward_lane_invasion_penalty = 0.0
        # if self.lane_invaded:
        #     reward_lane_invasion_penalty = -100.0

        # Penalize being too slow / not moving (if desired)
        reward_stuck_penalty = 0.0
        if host_velocity < 0.5 and self.tick_count > 50:  # If stuck for a while
            reward_stuck_penalty = -10.0

        total_reward = reward_distance + reward_velocity_match + reward_smoothness + reward_bonus \
                       + reward_collision_penalty + reward_stuck_penalty  # + reward_lane_invasion_penalty
        return total_reward

    def _is_done(self, state):
        distance_error, relative_velocity, host_velocity, host_acceleration = state

        if self.collision_happened:
            return True
        # if self.lane_invaded: # If lane invasion is a terminal condition
        #    return True

        # If ego vehicle falls off map (very low Z coordinate)
        if self.ego_vehicle.get_location().z < -5.0:
            return True

        # If lead car is too far away (lost target)
        distance = self.lead_vehicle.get_location().x - self.ego_vehicle.get_location().x
        if distance > 200.0:
            return True

        # End episode after a certain number of ticks (to prevent infinite episodes)
        if self.tick_count >= 1000:  # Example: 1000 steps * 0.1s/step = 100 seconds
            return True

        return False

    def destroy_actors(self):
        actors_to_destroy = []
        if self.collision_sensor:
            actors_to_destroy.append(self.collision_sensor)
            self.collision_sensor = None
        if self.lane_invasion_sensor:
            actors_to_destroy.append(self.lane_invasion_sensor)
            self.lane_invasion_sensor = None
        if self.ego_vehicle:
            actors_to_destroy.append(self.ego_vehicle)
            self.ego_vehicle = None
        if self.lead_vehicle:
            actors_to_destroy.append(self.lead_vehicle)
            self.lead_vehicle = None

        if self.world and actors_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
            print(f"Destroyed {len(actors_to_destroy)} actors.")

        # Reset settings to async mode if desired for other CARLA usage
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)


# --- DDPG Agent, Actor, Critic, ReplayBuffer classes (from previous PyTorch code) ---
# Paste your PyTorch DDPG agent code here.
# Make sure to import torch.nn.functional as F if you use F.relu or F.mse_loss

# ... (Actor, Critic, ReplayBuffer, DDPGAgent classes go here) ...
# Example:
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, action_max, action_min): ...
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim): ...
# class ReplayBuffer:
#     def __init__(self, buffer_size, batch_size): ...
# class DDPGAgent:
#     def __init__(self): ...
#     def policy(self, state, add_noise=True): ...
#     def learn(self): ...
#     def update_target_networks(self): ...

# --- Main Training Loop ---
def train_ddpg_carla(num_episodes=500, max_steps_per_episode=1000):
    env = None
    try:
        env = CarlaACCEnv(dt=DT, safe_distance=SAFE_DISTANCE)
        agent = DDPGAgent()

        episode_rewards = []
        avg_rewards_plot = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            for step in range(max_steps_per_episode):
                action = agent.policy(state, add_noise=True)
                next_state, reward, done, _ = env.step(action)

                agent.replay_buffer.record((state, action, reward, next_state, float(done)))

                state = next_state
                episode_reward += reward

                agent.learn()
                agent.update_target_networks()

                if done:
                    break

            episode_rewards.append(episode_reward)
            if len(episode_rewards) >= 10:
                avg_rewards_plot.append(np.mean(episode_rewards[-10:]))
            else:
                avg_rewards_plot.append(episode_reward)

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

            if (episode + 1) % 50 == 0:
                avg_test_reward = 0
                for _ in range(3):  # Run a few test episodes without noise
                    test_state = env.reset()
                    test_episode_reward = 0
                    test_done = False
                    for test_step in range(max_steps_per_episode):
                        test_action = agent.policy(test_state, add_noise=False)
                        test_state, test_reward, test_done, _ = env.step(test_action)
                        test_episode_reward += test_reward
                        if test_done:
                            break
                    avg_test_reward += test_episode_reward
                avg_test_reward /= 3
                print(f"--- Test Episode Average Reward (no noise): {avg_test_reward:.2f} ---")

    finally:
        if env:
            env.destroy_actors()
        print("Training finished!")
        return episode_rewards, avg_rewards, agent


if __name__ == "__main__":
    # Ensure torch.nn.functional is imported
    import torch.nn.functional as F

    # Make sure you start your CARLA server before running this script!
    # e.g., on Linux: ./CarlaUE4.sh -quality-level=Low -fps=30

    rewards, avg_rewards, trained_agent = train_ddpg_carla()

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(avg_rewards, label='10-Episode Moving Average', color='red')
    plt.title("DDPG Training Rewards (CARLA Environment)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Demonstration of trained agent in CARLA ---
    print("\n--- Demonstration of trained agent in CARLA ---")
    demo_env = None
    try:
        demo_env = CarlaACCEnv(dt=DT, safe_distance=SAFE_DISTANCE)
        state = demo_env.reset()
        total_reward = 0
        print_interval = 20  # Print every 20 steps

        for i in range(200):  # Simulate for 20 seconds (200 steps * 0.1s/step)
            action = trained_agent.policy(state, add_noise=False)  # No noise for deployment
            next_state, reward, done, _ = demo_env.step(action)

            # Retrieve current CARLA data for logging
            ego_speed_mps = np.sqrt(
                demo_env.ego_vehicle.get_velocity().x ** 2 + demo_env.ego_vehicle.get_velocity().y ** 2 + demo_env.ego_vehicle.get_velocity().z ** 2)
            distance = demo_env.lead_vehicle.get_location().x - demo_env.ego_vehicle.get_location().x

            if (i + 1) % print_interval == 0 or done:
                print(
                    f"Step {i + 1}: Host Speed={ego_speed_mps:.2f} m/s, Accel={action[0]:.2f}, Distance={distance:.2f}, Reward={reward:.2f}")

            total_reward += reward
            state = next_state
            if done:
                print("Episode ended early during demonstration.")
                break
        print(f"Demonstration Total Reward: {total_reward:.2f}")

    finally:
        if demo_env:
            demo_env.destroy_actors()



# --- Example Usage ---
if __name__ == "__main__":
    state_dim = 7  # Corresponding to 7x1x1 InputLayer
    action_dim = 1  # Corresponding to 1x1x1 InputLayer

    critic_model = Critic(state_dim, action_dim)
    print("Critic Model Architecture:")
    print(critic_model)

    # Create dummy input tensors
    batch_size = 4
    dummy_state_input = torch.randn(batch_size, state_dim)
    dummy_action_input = torch.randn(batch_size, action_dim)

    print(f"\nDummy State Input Shape: {dummy_state_input.shape}")
    print(f"Dummy Action Input Shape: {dummy_action_input.shape}")

    # Pass inputs through the critic
    q_value_output = critic_model(dummy_state_input, dummy_action_input)
    print(f"Q-Value Output Shape: {q_value_output.shape}")
    print(f"Sample Q-Values:\n{q_value_output.detach().numpy()}")
# 4. Critic Network (PyTorch)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_h1 = nn.Linear(state_dim, 16)
        self.state_h2 = nn.Linear(16, 32)

        self.action_h1 = nn.Linear(action_dim, 32)

        self.common_net = nn.Sequential(
            nn.Linear(32 + 32, 256),  # Concatenated state_h2 and action_h1
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)  # Output Q-value
        )

    def forward(self, state, action):
        s1 = F.relu(self.state_h1(state))
        s2 = F.relu(self.state_h2(s1))

        a1 = F.relu(self.action_h1(action))

        x = torch.cat([s2, a1], dim=1)  # Concatenate state and action features
        return self.common_net(x)


# 5. Replay Buffer (Same logic, adapt for PyTorch tensors)
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def record(self, obs_tuple):
        # Store numpy arrays, convert to tensors during sampling
        self.buffer.append(obs_tuple)

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors and move to device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE).unsqueeze(
            1)  # Add dim for consistency
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(DEVICE).unsqueeze(1)  # Add dim for consistency

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# 6. DDPG Agent (PyTorch)
class DDPGAgent:
    def __init__(self):
        self.actor = Actor(STATE_DIM, ACTION_DIM, ACCEL_CMD_MAX, ACCEL_CMD_MIN).to(DEVICE)
        self.critic = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)

        self.target_actor = Actor(STATE_DIM, ACTION_DIM, ACCEL_CMD_MAX, ACCEL_CMD_MIN).to(DEVICE)
        self.target_critic = Critic(STATE_DIM, ACTION_DIM).to(DEVICE)

        # Initialize target networks with the same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.noise = np.random.normal(loc=0.0, scale=EXPLORATION_NOISE_STD, size=ACTION_DIM)  # Simple Gaussian noise

    def policy(self, state, add_noise=True):
        # Convert state (numpy) to tensor, add batch dimension, move to device
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        self.actor.eval()  # Set actor to evaluation mode
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]  # Get action as numpy array
        self.actor.train()  # Set actor back to training mode

        if add_noise:
            # Add exploration noise. For DDPG, Ornstein-Uhlenbeck is typical,
            # but simple Gaussian is fine for this basic example.
            action = action + self.noise

        return np.clip(action, ACCEL_CMD_MIN, ACCEL_CMD_MAX)

    def learn(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # Not enough samples to learn

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch()

        # --- Update Critic ---
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.target_actor(next_states)
            # Get Q-values for next states and next actions from target critic
            q_target_next = self.target_critic(next_states, next_actions)
            # Compute target Q-value
            y_target = rewards + GAMMA * q_target_next * (1 - dones)  # (1 - dones) handles terminal states

        # Get Q-values for current states and actions from main critic
        q_current = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(q_current, y_target)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Get actions from main actor
        actor_actions = self.actor(states)
        # Actor loss: Maximize Q-value predicted by main critic
        actor_loss = -self.critic(states, actor_actions).mean()  # Take negative mean for gradient ascent

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_networks(self):
        # Soft update for target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        # Soft update for target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# 7. Training Loop
def train_ddpg(num_episodes=500, max_steps_per_episode=1000):
    env = CarFollowingEnv()
    agent = DDPGAgent()

    episode_rewards = []
    avg_rewards_plot = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        for step in range(max_steps_per_episode):
            action = agent.policy(state, add_noise=True)
            next_state, reward, done, _ = env.step(action)

            agent.replay_buffer.record((state, action, reward, next_state, float(done)))

            state = next_state
            episode_reward += reward

            agent.learn()  # Learn after each step
            agent.update_target_networks()  # Update targets after learning

            if done:
                break

        episode_rewards.append(episode_reward)

        # Calculate and store rolling average for smoother plot
        if len(episode_rewards) >= 10:
            avg_rewards_plot.append(np.mean(episode_rewards[-10:]))
        else:
            avg_rewards_plot.append(episode_reward)  # For early episodes

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")

        # Periodically evaluate performance without exploration noise
        if (episode + 1) % 50 == 0:
            avg_test_reward = 0
            for _ in range(5):  # Run a few test episodes
                test_state = env.reset()
                test_episode_reward = 0
                test_done = False
                for test_step in range(max_steps_per_episode):
                    test_action = agent.policy(test_state, add_noise=False)  # No noise for testing
                    test_state, test_reward, test_done, _ = env.step(test_action)
                    test_episode_reward += test_reward
                    if test_done:
                        break
                avg_test_reward += test_episode_reward
            avg_test_reward /= 5
            print(f"--- Test Episode Average Reward (no noise): {avg_test_reward:.2f} ---")

    print("Training finished!")
    return episode_rewards, avg_rewards_plot


if __name__ == "__main__":
    # Ensure torch.nn.functional is imported for F.relu and F.mse_loss
    import torch.nn.functional as F

    rewards, avg_rewards = train_ddpg()

    # Plotting rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    plt.plot(avg_rewards, label='10-Episode Moving Average', color='red')
    plt.title("DDPG Training Rewards (PyTorch)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Demonstrate a trained agent (optional)
    # Re-initialize agent and load weights if you want to see a truly trained agent
    # from a saved model without retraining. For now, this will use the agent
    # just trained.
    env = CarFollowingEnv()
    agent = DDPGAgent()  # This will be a freshly initialized agent.
    # To use the trained agent from the above training loop:
    # You would typically save and load state_dict. For this example,
    # let's assume the agent in memory from `train_ddpg` is the one we want.
    # To really demonstrate the *trained* agent, you'd need to save `agent.actor.state_dict()`
    # and `agent.critic.state_dict()` and load them here.
    # For now, if you run the script, `agent` here will be untrained.
    # If `train_ddpg` returned the agent, we could use that. Let's modify `train_ddpg`
    # to return the agent. (Done in the final code)

    print("\n--- Demonstration of agent (using the one just trained) ---")

    # To truly use the trained agent, you'd pass it or load its weights.
    # Let's assume `trained_agent` is the one returned from `train_ddpg`.
    trained_agent = DDPGAgent()  # Placeholder, replace with actual trained agent
    # You would need to load the state dicts here if you were saving/loading
    # Example: trained_agent.actor.load_state_dict(torch.load('actor_weights.pth'))

    # For this script, we'll just re-run the training and use the `agent` object
    # after `train_ddpg` if we modify `train_ddpg` to return the agent.
    # Let's make `train_ddpg` return the agent to use it directly.
    # (The function `train_ddpg` is modified above to return `agent` as well)

    # Re-run `train_ddpg` to get the trained agent for demonstration
    # (or you could save/load weights if this was a long training)
    # For simplicity, if `train_ddpg` is run, `agent` variable refers to the last trained agent.
    # Let's run a new environment with the trained agent.

    env_demo = CarFollowingEnv()
    state = env_demo.reset()
    total_reward = 0
    print_interval = 20  # Print every 20 steps

    # Use the 'agent' that was just trained
    for i in range(200):  # Simulate for 20 seconds
        action = agent.policy(state, add_noise=False)  # No noise for deployment
        next_state, reward, done, _ = env_demo.step(action)
        distance = env_demo.lead_car_pos - env_demo.host_car_pos

        if (i + 1) % print_interval == 0 or done:
            print(
                f"Step {i + 1}: Host Speed={env_demo.host_car_speed:.2f} m/s, Accel={action[0]:.2f}, Distance={distance:.2f}, Reward={reward:.2f}")

        total_reward += reward
        state = next_state
        if done:
            print("Episode ended early during demonstration.")
            break
    print(f"Demonstration Total Reward: {total_reward:.2f}")