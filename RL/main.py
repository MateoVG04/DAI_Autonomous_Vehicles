
import time
import logging
from stable_baselines3 import TD3
from train import RemoteCarlaEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

def evaluate(model_path, env, episodes, max_steps):
    # Load environment & model
    model = TD3.load(model_path, env=env)

    mean_reward = 0.0
    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0

        logger.log(logging.INFO, f"=== Episode {ep + 1} ===")

        for step in range(max_steps):
            # deterministic=True → no exploration noise during evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward

            # time.sleep(0.02)  # 50 ms = ~20 FPS, adjust as needed

            if terminated or truncated:
                break

        logger.log(logging.INFO, f"Episode reward: {ep_reward}")
        mean_reward += ep_reward

    logger.log(logging.INFO,f"mean reward over {episodes} is {mean_reward/episodes}")

    env.close()
    logger.log(logging.INFO, "Evaluation finished.")


if __name__ == "__main__":
    env = RemoteCarlaEnv()
    start = time.time()
    evaluate("ddpg_carla_final", env, episodes=5, max_steps=1000)
    end = time.time()
    logger.log(logging.INFO, f"Total evaluation time: {(end - start) / 60} minutes")