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
        self.remote_env = Pyro4.Proxy("PYRO:carla.environment@localhost:45865") # for now, hardcoded port

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


env = RemoteCarlaEnv()

model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./ddpg_carla_tensorboard/")
model.learn(total_timesteps=50000)

model.save("ddpg_carla_model")

print("finished training")