import os
import numpy as np
import time
import wandb
import logging
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
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

def load_create_model(env: RemoteCarlaEnv, model_path: str = None) -> TD3:
    """
    Load an existing TD3 model or create a new one.
    :param env: Carla Gym environment.
    :param model_path: Path to the saved model file.
    :return: TD3 model.
    """
    n_actions = env.action_space.shape[0]

    # for exploration, noise = OU noise
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1)

    # Check if model exists
    if model_path and os.path.exists(model_path):
        logger.log(logging.INFO, f"Loading model from {model_path}...")
        model = TD3.load(model_path, env=env, action_noise=action_noise)
        logger.log(logging.INFO, "Model loaded.")
    # Else create a new one
    else:
        logger.log(logging.INFO, "Creating new TD3 model...")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=256,
            tau=0.005,
            action_noise=action_noise,
            verbose=1,
            buffer_size=200_000,
        )
        logger.log(logging.INFO, "New model created.")

    return model

def wandb_setup(timesteps: int = 100_000) -> wandb.sdk.wandb_run.Run:
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

def train(env: RemoteCarlaEnv, model_path: str, timesteps: int, run=None) -> None:
    """
    Train the TD3 agent on the given environment.
    :param env: Carla Gym environment.
    :param timesteps: Amount of training timesteps.
    :param run: Wandb run object for logging.
    :return: None
    """

    save_path = f"./models/{run.id}"
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=save_path,
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # 3. Setup WandB Callback
    wandb_callback = WandbCallback(
        gradient_save_freq=5000,
        model_save_path=f"./models/{run.id}",
        verbose=2
    )

    callbacks = CallbackList([checkpoint_callback, wandb_callback])

    model = load_create_model(env, model_path=model_path)

    logger.log(logging.INFO, f"Training...")
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=False,

    )
    logger.log(logging.INFO,"Training finished.")

    model.save(f"./models/carla_model_{timesteps}_multiMap")
    logger.log(logging.INFO, "Model saved.")

if __name__ == '__main__':
    timesteps = 500_000
    env = RemoteCarlaEnv()
    run = wandb_setup(timesteps=timesteps)
    start = time.time()
    train(env=env, timesteps=timesteps, model_path=f"./models/carla_model_{timesteps}_multiMap", run=run)
    end = time.time()
    logger.log(logging.INFO, f"Training time: {(end - start) // 60} minutes")