import gymnasium as gym
import Pyro4
import numpy as np
import logging
import time
import serpent


"""
Carla Remote Environment accessed via Pyro4.
"""

logger = logging.getLogger(__name__)
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
        logger.info("Checking remote connection...")
        try:
            dummy_obs, _ = self.remote_env.reset()
            obs_dim = len(dummy_obs)
        except:
            obs_dim = 35  # Fallback if server isn't up yet
        logger.info( f"Detected Remote Observation Dim: {obs_dim}")

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def connect(self):
        """Attempts to connect to the server, retrying indefinitely."""
        while True:
            try:
                self.remote_env = Pyro4.Proxy("PYRONAME:carla.environment")
                self.remote_env._pyroBind() # Test connection
                logger.info( "Connected to CARLA Server.")
                return
            except Exception:
                logger.info("Waiting for CARLA Server...")
                time.sleep(5)

    def reset(self, seed=None, options=None):
        """Resets the remote environment and returns the initial observation."""
        try:
            obs_list, info = self.remote_env.reset()
            obs = np.array(obs_list, dtype=np.float32)

            return obs, info
        except (Pyro4.errors.ConnectionClosedError, Pyro4.errors.CommunicationError):
            logger.warning("Connection lost during RESET. Waiting for server restart...")
            self.connect()


    def step(self, action: list):
        """Takes a step in the remote environment using the provided action."""
        try:
            action = float(np.array(action).squeeze())
            obs, reward, terminated, truncated, info = self.remote_env.step(action)

            return np.array(obs, dtype=np.float32), reward, terminated, truncated, info
        except Exception:
            logger.warning("Connection lost during STEP. Waiting for server restart...")
            self.connect()
            return self.reset()

    def close(self):
        try:
            self.remote_env.close()
        except:
            pass

    def get_latest_image(self):
        img_bytes, shape, frame_id = self.remote_env.get_latest_image()

        if img_bytes is None:
            return None, None

        # Pyro+Serpent may have encoded bytes as a dict; convert it back.
        img_bytes = serpent.tobytes(img_bytes)  # handles dict/str/bytes

        h, w, c = shape
        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, c))

        return img_np, frame_id

    def get_latest_lidar_points(self):
        raw, shape = self.remote_env.get_latest_lidar_points()
        if raw is None or shape is None:
            return None

        # Same issue as image bytes: ensure raw is real bytes
        raw = serpent.tobytes(raw)

        pts = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return pts

    def draw_detections(self, detections, img_width=800, img_height=600):
        """
        Forward YOLO detections to remote CarlaEnv for debug drawing.

        detections: list of dicts with keys:
            - "name": str
            - "conf": float
            - "bbox": [x1, y1, x2, y2]
        """
        # Make sure everything is built-in types (no numpy scalars)
        safe_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            safe_dets.append({
                "name": str(det["name"]),
                "conf": float(det["conf"]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        # Remote call – server will do world.debug drawing
        self.remote_env.draw_detections(safe_dets, int(img_width), int(img_height))

