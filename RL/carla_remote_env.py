import gymnasium as gym
import Pyro4
import numpy as np
import logging


"""
Carla Remote Environment accessed via Pyro4.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
    """
    Gym environment that connects to a remote Carla environment via Pyro4.
    Action space: Continuous [-1, 1] for steering control.
    Observation space: Continuous vector received from the remote environment.
    """
    def __init__(self):
        super().__init__()

        # Establish Pyro4 connection to remote Carla environment
        self.remote_env = Pyro4.Proxy("PYRONAME:carla.environment")

        # Test connection and get observation dimension
        logger.log(logging.INFO,"Checking remote connection...")
        dummy_obs, _ = self.remote_env.reset()
        obs_dim = len(dummy_obs)
        logger.log(logging.INFO, f"Detected Remote Observation Dim: {obs_dim}")

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def step(self, action: list):

        action = float(np.array(action).squeeze())
        obs, reward, terminated, truncated, info = self.remote_env.step(action)

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_list, info = self.remote_env.reset()
        obs = np.array(obs_list, dtype=np.float32)

        return obs, info

    def close(self):
        try:
            self.remote_env.close()
        except:
            pass