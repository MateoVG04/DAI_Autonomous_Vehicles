import gymnasium as gym
import Pyro4
import numpy as np
import logging
import time


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
        try:
            dummy_obs, _ = self.remote_env.reset()
            obs_dim = len(dummy_obs)
        except:
            obs_dim = 35  # Fallback if server isn't up yet
        logger.log(logging.INFO, f"Detected Remote Observation Dim: {obs_dim}")

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def connect(self):
        """Attempts to connect to the server, retrying indefinitely."""
        while True:
            try:
                self.remote_env = Pyro4.Proxy("PYRONAME:carla.environment")
                self.remote_env._pyroBind() # Test connection
                logger.log(logging.INFO, "✅ Connected to CARLA Server.")
                return
            except (Exception):
                logger.log(logging.INFO,"Waiting for CARLA Server...", end="\r")
                time.sleep(5)

    def reset(self, seed=None, options=None):
        try:
            obs_list, info = self.remote_env.reset()
            obs = np.array(obs_list, dtype=np.float32)

            return obs, info
        except (Pyro4.errors.ConnectionClosedError, Pyro4.errors.CommunicationError):
            logger.log(logging.WARNING,"Connection lost during RESET. Waiting for server restart...")
            self.connect()


    def step(self, action: list):
        try:
            action = float(np.array(action).squeeze())
            obs, reward, terminated, truncated, info = self.remote_env.step(action)

            return np.array(obs, dtype=np.float32), reward, terminated, truncated, info
        except:
            logger.log(logging.WARNING,"Connection lost during STEP. Waiting for server restart...")
            self.connect()
            return self.reset()

    def close(self):
        try:
            self.remote_env.close()
        except:
            pass