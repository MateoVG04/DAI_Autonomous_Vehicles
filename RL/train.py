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


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["WANDB_API_KEY"] = "232f438f252e30a2b8726b6acc2920339a1bbadd"


def get_latest_checkpoint(run_id):
    """Finds the newest zip file in ./models/{run_id}/"""
    ckpt_dir = f"./models/{run_id}"
    if not os.path.exists(ckpt_dir): return None

    files = glob.glob(f"{ckpt_dir}/*.zip")
    return max(files, key=os.path.getctime) if files else None


def train(total_timesteps, save_freq, prev_run_id):

    # 1. INITIALIZE WANDB AND ENV
    env = RemoteCarlaEnv()
    run = wandb.init(
        project="carla-rl",
        id=prev_run_id,
        resume="allow",
        config={"policy": "TD3", "timesteps": total_timesteps},
        sync_tensorboard=True
    )

    # 2. PATHS
    checkpoint_dir = f"./models/{run.id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. LOAD/CREATE MODEL
    latest_ckpt = get_latest_checkpoint(run.id)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # If model load
    if latest_ckpt:
        logger.info(f"RESUMING from checkpoint: {latest_ckpt}")
        model = TD3.load(latest_ckpt, env=env)
        model.action_noise = action_noise  # Re-attach noise

        # Load Buffer
        buffer_path = latest_ckpt.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(buffer_path):
            logger.info("Loading Replay Buffer...")
            model.load_replay_buffer(buffer_path)

    # Else create model
    else:
        logger.info(f"Starting FRESH training run: {run.id}")
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

    # 4. CALLBACK FOR CHECKPOINT AND WANDB
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix="td3_ckpt",
            save_replay_buffer=True,  # Saves the heavy buffer for crash recovery
            save_vecnormalize=True
        ),
        WandbCallback(gradient_save_freq=5000, verbose=2)
    ])

    # 5. TRAINING & SAVE
    try:
        logger.info("Training started...")
        start_time = time.time()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            reset_num_timesteps=False  # Keeps the progress bar correct on resume
        )

        # Save model after training
        logger.info("Training Finished.")
        final_path = f"td3_carla_2_{TOTAL_TIMESTEPS}"
        model.save(final_path)
        logger.info(f"Final model saved to: {os.path.abspath(final_path)}.zip")

    # Save early if exception
    except Exception:
        logger.warning("Interrupt detected. Saving emergency checkpoint...")
        model.save(f"{checkpoint_dir}/model_interrupted")
        model.save_replay_buffer(f"{checkpoint_dir}/buffer_interrupted.pkl")

    # Dont save if keyboard interrupt (user realised bad training)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected. Saving nothing...")

    # Finish training by closing env and stopping run
    finally:
        run.finish()
        logger.info(f"Total Time: {(time.time() - start_time) // 60:.0f} minutes")


if __name__ == '__main__':
    TOTAL_TIMESTEPS = 100_000
    SAVE_FREQ = 25_000  # Save buffer
    PREVIOUS_RUN_ID = "y8tmpvh4"  # Set this string (e.g. "a1b2c3d4") to resume a crash
    # Hardy elevator
    train(TOTAL_TIMESTEPS, SAVE_FREQ, PREVIOUS_RUN_ID)