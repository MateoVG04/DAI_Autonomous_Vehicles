import os
import glob
import time
import wandb
import logging
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback
from carla_remote_env import RemoteCarlaEnv

# --- Configuration ---
TOTAL_TIMESTEPS = 500_000
SAVE_FREQ = 100_000  # Save buffer & model every 100k steps
PREVIOUS_RUN_ID = None # Set this string (e.g. "a1b2c3d4") to resume a crash
WANDB_KEY = "232f438f252e30a2b8726b6acc2920339a1bbadd"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["WANDB_API_KEY"] = WANDB_KEY


def get_latest_checkpoint(run_id):
    """Finds the newest zip file in ./models/{run_id}/"""
    ckpt_dir = f"./models/{run_id}"
    if not os.path.exists(ckpt_dir): return None

    files = glob.glob(f"{ckpt_dir}/*.zip")
    return max(files, key=os.path.getctime) if files else None


def train():
    # 1. Initialize WandB & Environment
    env = RemoteCarlaEnv()
    run = wandb.init(
        project="carla-rl",
        id=PREVIOUS_RUN_ID,
        resume="allow",
        config={"policy": "TD3", "timesteps": TOTAL_TIMESTEPS},
        sync_tensorboard=True
    )

    # 2. Paths
    checkpoint_dir = f"./models/{run.id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Load or Create Model
    latest_ckpt = get_latest_checkpoint(run.id)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if latest_ckpt:
        logger.info(f"🔄 RESUMING from checkpoint: {latest_ckpt}")
        model = TD3.load(latest_ckpt, env=env)
        model.action_noise = action_noise  # Re-attach noise

        # Load Buffer (Critical for TD3 continuity)
        buffer_path = latest_ckpt.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            logger.info("📦 Loading Replay Buffer...")
            model.load_replay_buffer(buffer_path)
    else:
        logger.info(f"🆕 Starting FRESH training run: {run.id}")
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=f"./runs/{run.id}"
        )

    # 4. Callbacks (Save every 100k steps)
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=checkpoint_dir,
            name_prefix="td3_ckpt",
            save_replay_buffer=True,  # Saves the heavy buffer for crash recovery
            save_vecnormalize=True
        ),
        WandbCallback(gradient_save_freq=5000, verbose=2)
    ])

    # 5. Start Training
    try:
        logger.info("Training started...")
        start_time = time.time()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False  # Keeps the progress bar correct on resume
        )

        # 6. Save Final Model to Root Folder
        logger.info("Training Finished.")
        final_path = f"td3_carla_{TOTAL_TIMESTEPS}"
        model.save(final_path)
        logger.info(f"Final model saved to: {os.path.abspath(final_path)}.zip")

    except Exception:
        logger.warning("Interrupt detected. Saving emergency checkpoint...")
        model.save(f"{checkpoint_dir}/model_interrupted")
        model.save_replay_buffer(f"{checkpoint_dir}/buffer_interrupted.pkl")
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected. Saving nothing...")

    finally:
        env.close()
        run.finish()
        logger.info(f"Total Time: {(time.time() - start_time) // 60:.0f} minutes")


if __name__ == '__main__':
    train()