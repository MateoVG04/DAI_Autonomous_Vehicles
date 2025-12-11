"""
First run carla_server.py with Python 3.8. shared venv
Second run this file with Python 3.12 shared venv
Lastly, look at the vnc http://localhost:6082/vnc.html
"""

import time
import logging
from stable_baselines3 import TD3
from ultralytics import YOLO

from RL.carla_remote_env import RemoteCarlaEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(stream_handler)


def main(env:RemoteCarlaEnv, RL_model_path: str, obj_model_path: str, max_steps: int) -> None:
    """
    Main loop to run the trained TD3 agent in the remote CARLA environment,
    combined with YOLO object detection.

    This is not an evaluation function; it just runs one route (from a random
    start to a random destination as defined by the env) until termination.
    """
    model = TD3.load(RL_model_path, env=env)
    obj_detect_model = YOLO(obj_model_path)

    obs, info = env.reset()
    ep_reward = 0.0
    logger.info("=== Run started ===")

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        img, frame_id = env.get_latest_image()
        if img is not None:
            results = obj_detect_model.predict(img, verbose=False)
            result = results[0]
            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                cls_list = boxes.cls.tolist()
                conf_list = boxes.conf.tolist()

                for cls, conf in zip(cls_list, conf_list):
                    name = result.names[int(cls)]
                    detections.append({
                        "name": name,
                        "conf": float(conf),
                    })

            if detections:
                env.draw_detections(detections)

        # Optional: slow down for visualization
        # time.sleep(0.01)

        if terminated or truncated:
            logger.info(
                f"Run finished after {step + 1} steps "
                f"(terminated={terminated}, truncated={truncated})"
            )
            break

    logger.info(f"Total reward this run: {ep_reward:.2f}")
    env.close()
    logger.info("CARLA run finished.")


if __name__ == "__main__":
    env = RemoteCarlaEnv()
    start = time.time()
    main(env,"./RL/Model_TD3/td3_carla_500000", "./Machine_Vision/runs/Train5/best.pt", max_steps=100_000)
    end = time.time()
    logger.info(f"Total wall time: {(end - start) / 60:.2f} minutes")
