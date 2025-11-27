import gymnasium as gym
import Pyro4
import numpy as np
import time
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Connect to the remote object published by the server
        # change port
        self.remote_env = Pyro4.Proxy("PYRONAME:carla.environment") # for now, hardcoded port

        # define spaces due to serialization issues
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(35,), dtype=np.float32)

    def step(self, action):
        action = float(np.array(action).squeeze())
        obs, reward, terminated, truncated, info = self.remote_env.step(action)
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_list, info = self.remote_env.reset()
        print("OBS SHAPE:", len(obs_list))
        print("OBS VALUES:", obs_list)
        obs = np.array(obs_list, dtype=np.float32)
        return obs, info

    def close(self):
        try:
            self.remote_env.close()
        except:
            pass

def train(env, timesteps):
    n_actions = env.action_space.shape[0]
    # for exploration, noise = OU noise
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),  # throttle/brake exploration
        theta=0.1  # smoothness of changes
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,               # soft update
        action_noise=action_noise,
        verbose=1,
        buffer_size=500_000,
        tensorboard_log="./ddpg_carla_tensorboard/"
    )

    # Save checkpoints every 20k steps
    # checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path="./checkpoints/", name_prefix="ddpg_carla")

    model.learn(timesteps)
    print("Training finished.")

    model.save("ddpg_carla_final")
    print("Model saved.")

if __name__ == '__main__':
    env = RemoteCarlaEnv()
    start = time.time()
    train(env, 100_000)
    end = time.time()
    print("Training time:", (end - start) // 60, "minutes")