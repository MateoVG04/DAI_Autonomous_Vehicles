import os
import numpy as np
import time
import wandb
import logging
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from carla_remote_env import RemoteCarlaEnv

"""
Train a TD3 agent on a remote Carla environment via Pyro4.
A remote Carla environment server must be running before executing this script.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

os.environ["WANDB_API_KEY"] = "232f438f252e30a2b8726b6acc2920339a1bbadd"

def wandb_setup(timesteps=100_000) -> wandb.sdk.wandb_run.Run:
    """
    Setup wandb logging for the training run.
    :param timesteps: Number of training timesteps.
    :return: wandb run object.
    """
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

def train(env: RemoteCarlaEnv, timesteps: int, run=None) -> None:
    """
    Train the TD3 agent on the given environment.
    :param env: Carla Gym environment.
    :param timesteps: Amount of training timesteps.
    :param run: Wandb run object for logging.
    :return: None
    """
    n_actions = env.action_space.shape[0]

    # for exploration, noise = OU noise
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1)

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