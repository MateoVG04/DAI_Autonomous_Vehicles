
import gymnasium as gym
import Pyro4
from stable_baselines3 import DDPG
import numpy as np


# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Connect to the remote object published by the server
        # change port
        self.remote_env = Pyro4.Proxy("PYRO:carla.environment@localhost:46833") # for now, hardcoded port

        # define spaces due to serialization issues
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                            high=np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0,
                                            shape=(18,), dtype=np.float32)

    def step(self, action):

        obs, reward, done, truncated, info = self.remote_env.step(action.tolist())
        return np.array(obs), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # 5. Forward the 'reset' call
        obs_list, info = self.remote_env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, info

    def close(self):
        self.remote_env.close()


env = RemoteCarlaEnv()

model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

