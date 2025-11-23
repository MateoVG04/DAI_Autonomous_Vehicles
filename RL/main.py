
import time
from stable_baselines3 import TD3
from train import RemoteCarlaEnv

def evaluate(model_path, env, episodes, max_steps):
    # Load environment & model
    model = TD3.load(model_path, env=env)

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0

        print(f"=== Episode {ep + 1} ===")

        for step in range(max_steps):
            # deterministic=True → no exploration noise during evaluation
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            # --- Debug vehicle control ---
            print(info)


            ep_reward += reward

            time.sleep(0.02)  # 50 ms = ~20 FPS, adjust as needed

            if terminated or truncated:
                break

        print(f"Episode reward: {ep_reward}")

    env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    env = RemoteCarlaEnv()
    start = time.time()
    evaluate("ddpg_carla_final", env, episodes=5, max_steps=100)
    end = time.time()
    print(f"Total evaluation time: {end - start}")