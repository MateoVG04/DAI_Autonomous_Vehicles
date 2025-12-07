import os
import glob
import numpy as np
import time
import wandb
import logging
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from carla_remote_env import RemoteCarlaEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

os.environ["WANDB_API_KEY"] = "232f438f252e30a2b8726b6acc2920339a1bbadd"

def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recent zip file in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    # Get list of all zip files in the directory
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))

    if not list_of_files:
        return None

    # Return the file with the latest creation time
    return max(list_of_files, key=os.path.getctime)


def load_create_model(env: RemoteCarlaEnv, run_id: str, buffer_size: int = 200_000) -> TD3:
    """
    Load the latest existing TD3 model or create a new one.
    """
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Directory where checkpoints are saved for this specific run
    checkpoint_dir = f"./models/{run_id}"
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        logger.info(f"RESUMING: Found checkpoint: {latest_checkpoint}")

        # Load the model weights
        # We pass 'env' to ensure it re-binds to the remote environment
        model = TD3.load(latest_checkpoint, env=env)

        # Re-attach the action noise (it often gets lost in saving/loading)
        model.action_noise = action_noise

        # Load the Replay Buffer (Crucial for continuity)
        # CheckpointCallback saves buffer as "name_prefix_steps_replay_buffer.pkl"
        buffer_path = latest_checkpoint.replace(".zip", "_rb.pkl")

        if os.path.exists(buffer_path):
            logger.info(f"Loading Replay Buffer from {buffer_path}...")
            model.load_replay_buffer(buffer_path)
            logger.info(f"Replay Buffer Loaded. Training can continue seamlessly.")
        else:
            logger.warning(f"Replay Buffer NOT found. Agent will have to refill memory.")

    else:
        logger.info(f"No checkpoints found for run {run_id}. Creating NEW model...")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=256,
            tau=0.005,
            action_noise=action_noise,
            verbose=1,
            buffer_size=buffer_size,
            # Ensure Tensorboard logs continue in the same folder
            tensorboard_log=f"./runs/{run_id}"
        )

    return model


def wandb_setup(timesteps: int = 100_000, run_id: str = None) -> wandb.sdk.wandb_run.Run:
    """
    Setup wandb logging.
    If run_id is provided, it attempts to RESUME that specific run on the cloud.
    """
    run = wandb.init(
        project="carla-rl",
        id=run_id,  # If this is not None, we resume the same graph
        resume="allow",  # Allows resuming if id exists
        config={
            "policy": "TD3",
            "timesteps": timesteps,
            "env": "CarlaEnv",
        },
        sync_tensorboard=True,
        save_code=True,
    )
    return run


def train(env: RemoteCarlaEnv, timesteps: int, existing_run_id: str = None) -> None:
    # 1. Setup WandB (Resume if ID is passed)
    run = wandb_setup(timesteps, run_id=existing_run_id)

    # 2. Define Checkpoint Path based on Run ID
    # This keeps saves organized: ./models/abc1234/
    save_path = f"./models/{run.id}"
    os.makedirs(save_path, exist_ok=True)

    # 3. Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=save_path,
        name_prefix="td3_carla",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=25_000,
        model_save_path=f"./models/{run.id}",
        verbose=2
    )

    callbacks = CallbackList([checkpoint_callback, wandb_callback])

    # 4. Load or Create Model
    # This checks ./models/{run.id}/ for zip files
    model = load_create_model(env, run.id, buffer_size=200_000)

    logger.info(f"Training started for {timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False
        )

        logger.info("Training finished.")
        model.save(f"{save_path}/carla_model_final")

    except KeyboardInterrupt or Exception:
        logger.info("Training interrupted manually. Saving current state...")
        model.save(f"{save_path}/carla_model_interrupted")
        if model.replay_buffer:
            model.save_replay_buffer(f"{save_path}/carla_model_interrupted_replay_buffer.pkl")


if __name__ == '__main__':
    PREVIOUS_RUN_ID = "h6c4j1wl"  # e.g., "dcany1qd"

    TOTAL_TIMESTEPS = 150_000

    env = RemoteCarlaEnv()
    start = time.time()

    train(env=env, timesteps=TOTAL_TIMESTEPS, existing_run_id=PREVIOUS_RUN_ID)

    end = time.time()
    logger.info(f"Training time: {(end - start) // 60} minutes")