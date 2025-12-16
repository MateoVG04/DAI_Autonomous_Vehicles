import threading
import numpy as np
import Pyro4

@Pyro4.expose
class PyroStateServer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_lidar_result = {}

    def update_lidar_result(self, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray):
        with self.lock:
            self.latest_lidar_result = {
                "bboxes": bboxes.tolist() if isinstance(bboxes, np.ndarray) else bboxes,
                "labels": labels.tolist() if isinstance(labels, np.ndarray) else labels,
                "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            }

    def get_latest_lidar_result(self):
        with self.lock:
            return self.latest_lidar_result.copy()
