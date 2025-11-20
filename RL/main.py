import gymnasium as gym
import Pyro4
import numpy as np
import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Connect to the remote object published by the server
        # change port
        self.remote_env = Pyro4.Proxy("PYRO:carla.environment@localhost:33013") # for now, hardcoded port

        # define spaces due to serialization issues
        self.action_space = gym.spaces.Box(low=-1.0,high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(33,), dtype=np.float32)

    def step(self, action):
        action = float(np.array(action).squeeze())
        obs, reward, done, truncated, info = self.remote_env.step(action)
        return np.array(obs), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        obs_list, info = self.remote_env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, info

    def close(self):
        self.remote_env.close()


def train(env, total_timesteps):
    n_actions = env.action_space.shape[0]
    # for exploration, noise = OU noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.2 * np.ones(n_actions),  # throttle/brake exploration
        theta=0.15  # smoothness of changes
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,               # soft update
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./ddpg_carla_tensorboard/"
    )

    # Save checkpoints every 20k steps
    # checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path="./checkpoints/", name_prefix="ddpg_carla")

    model.learn(total_timesteps=total_timesteps)
    model.save("ddpg_carla_final")
    print("Training finished.")


def evaluate(model_path, env, episodes, max_steps):
    # Load environment & model
    model = DDPG.load(model_path, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0

        print(f"=== Episode {ep + 1} ===")

        for step in range(max_steps):
            # deterministic=True → no exploration noise during evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            time.sleep(0.05)  # 50 ms = ~20 FPS, adjust as needed

            if done or truncated:
                break

        print(f"Episode reward: {ep_reward}")

    env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    env = RemoteCarlaEnv()
    train(env, total_timesteps=500_000)
    evaluate("ddpg_carla_final", env, episodes=5, max_steps=10_000)