import random

import gymnasium as gym
import Pyro4
import numpy as np
import logging
import time
import serpent
import threading
import pygame

from runtime.pyro_state import PyroStateServer
#from runtime.pointpillars_lightweight.run.inference import CarlaWrapper
from runtime.CarlaWrapper import CarlaWrapper
from visualization.MinimalHUD import MinimalHUD
from runtime.Distance import DistanceSystem

"""
Carla Remote Environment accessed via Pyro4.
"""

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

def start_pyro_daemon(logger, state_obj, pyro_name: str, port=9090):
    """Starts the Pyro4 server in a background thread"""
    daemon = Pyro4.Daemon(host="0.0.0.0", port=port)
    uri = daemon.register(state_obj, pyro_name)
    logger.info(f"Pyro4 Server Running! Object URI: {uri}")
    daemon.requestLoop()

# Proxy class to interact with the remote Carla environment
class RemoteCarlaEnv(gym.Env):
    """
    Gym environment that connects to a remote Carla environment via Pyro4.
    Action space: Continuous [-1, 1] for steering control.
    Observation space: Continuous vector received from the remote environment.
    """
    def __init__(self):
        super().__init__()
        self.camera_width = 800
        self.camera_height = 600
        self.max_lidar_points = 120000
        self.hud_width = self.camera_width * 2  # double width for camera + LiDAR
        self.hud_height = self.camera_height * 2

        # Establish Pyro4 connection to remote Carla environment
        self.remote_env = Pyro4.Proxy("PYRONAME:carla.environment")
        # Pyro
        self.pyro_name = "pyrostateserver"
        self.pyro_port = random.randint(9100, 9200)
        self.pyro_state_server = PyroStateServer()
        self.pyro_thread = threading.Thread(target=start_pyro_daemon,
                                       args=(logger, self.pyro_state_server, self.pyro_name, self.pyro_port), daemon=True)
        self.pyro_thread.start()

        # Pygame
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.hud_width, self.hud_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        # pygame.display.set_caption("CARLA Simulation")
        pygame.display.set_caption("DAI - AlphaDrive")

        ## Setup shared memory
        self.shared_memory_filepath = "/dev/shm/carla_shared/carla_shared_v6.dat"
        self.shared_memory = CarlaWrapper(filename=self.shared_memory_filepath,
                                          image_width=self.camera_width,
                                          image_height=self.camera_height,
                                          max_lidar_points=self.max_lidar_points)

        self.hud = MinimalHUD(self.hud_width, self.hud_height, shared_memory=self.shared_memory,
                              pyro_state_server=self.pyro_state_server, logger=logger)

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

    def hud_logic(self, yolo_detections=None, distance_to_dest=0.0, dashboard_img=None):
        self.hud.tick()
        self.hud.render(self.display, vehicle=None, distance_to_dest=float(distance_to_dest), lane_dashboard=dashboard_img)

        if yolo_detections:
            self.hud.draw_yolo_overlay(yolo_detections, self.display)

        pygame.display.flip()

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


    def hud_tick(self):
        self.remote_env.hud_tick()

    def hud_render(self):
        self.remote_env.hud_render()

    def step(self, action: list):
        """Takes a step in the remote environment using the provided action."""
        try:
            action = float(np.array(action).squeeze())
            obs, reward, terminated, truncated, info = self.remote_env.step(action)

            return np.array(obs, dtype=np.float32), reward, terminated, truncated, info
        except Exception:
            logger.warning("Connection lost during STEP. Waiting for server restart...")
            self.connect()
            obs, info = self.reset()
            return obs, 0.0, True, True, info

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

    def set_distance(self, dist):
        """Passes the distance to the remote server via Pyro4"""
        try:
            self.remote_env.set_distance(float(dist))
        except Exception as e:
            logger.error(f"Failed to update distance: {e}")