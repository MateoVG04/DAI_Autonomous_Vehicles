import os

import gymnasium as gym
import Pyro4
import numpy as np
import time
import wandb
import logging
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

os.environ["WANDB_API_KEY"] = "232f438f252e30a2b8726b6acc2920339a1bbadd"

# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
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

    def step(self, action):
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

def wandb_setup(timesteps=100_000):
    run = wandb.init(
        project="carla-rl",
        config={
            "policy": "TD3",
            "timesteps": timesteps,
            "env": "CarlaEnv",
        },
        sync_tensorboard=True,
        save_code=True,
    )
    logger.log(logging.INFO, f"Wandb setup complete, run: {run.id}")
    return run

def train(env, timesteps, run=None):
    n_actions = env.action_space.shape[0]

    # for exploration, noise = OU noise
    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1)

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        action_noise=action_noise,
        verbose=1,
        buffer_size=100_000,
        tensorboard_log=f"./run/{run.id}/"
    )

    logger.log(logging.INFO, f"Training...")
    model.learn(
        total_timesteps=timesteps,
        callback=WandbCallback(
            gradient_save_freq=5000,
            model_save_path=f"./models/{run.id}",
            verbose=2
        ),

    )
    logger.log(logging.INFO,"Training finished.")

    model.save(f"carla_model_{timesteps}_2")
    logger.log(logging.INFO, "Model saved.")

if __name__ == '__main__':
    timesteps = 300_000
    env = RemoteCarlaEnv()
    run = wandb_setup(timesteps=timesteps)
    start = time.time()
    train(env=env, timesteps=timesteps, run=run)
    end = time.time()
    logger.log(logging.INFO, f"Training time: {(end - start) // 60} minutes")