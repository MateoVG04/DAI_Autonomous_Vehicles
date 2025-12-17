import logging
import sys
from collections import deque

from tensorboard.compat.tensorflow_stub.errors import InvalidArgumentError
from torch import Tensor

from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Deque
import Pyro4

import numpy as np
import os
import torch

class SharedMemoryArray:
    def __init__(self,
                 data_shape: List[int],
                 reserved_count: int,
                 datatype) -> None:
        """
        A definition for a type of data saved in the buffer file

        :param data_shape: Data shape to be converted into a linear size (12, 12, 1) -> 144 for example
        :param reserved_count: How many of those objects can be placed
        :param datatype: Size also depends on the datatype
        """
        self.data_shape = data_shape
        self.reserved_count = reserved_count
        self.datatype = datatype

        self.datatype_size = {np.uint8: 1, np.float32: 4, np.float64: 8}[datatype]

        self.current_write_index = 0

    @property
    def slot_size(self):
        return np.prod(self.data_shape) * self.datatype_size

    @property
    def reserved_size(self):
        """
        Size to be reserved in the buffer (in bytes)
        :return:
        """
        return self.slot_size * self.reserved_count


class SharedMemoryManager:
    def __init__(self, filename: str,
                 data_arrays: List[SharedMemoryArray],
                 ):
        """

        :param filename:
        """
        self.filepath_str: str = filename

        # ----- Data Arrays
        self.write_array = SharedMemoryArray(data_shape=[1], reserved_count=len(data_arrays), datatype=np.uint8)
        self.data_arrays = data_arrays

        # ----- Creating binds for use in this class
        # Making sure the file already exists
        path = Path(filename)
        self.init_file(path)

        # ---- Helper views
        # Main memory
        # | write_indices | ... all memory arrays |
        self._mm = np.memmap(filename, dtype=np.uint8, mode="r+", shape=(self.total_size,))
        # Write indices
        self._write_index_mm = np.ndarray(len(self.data_arrays), dtype=np.uint8,
                                          buffer=self._mm[:self.write_array.reserved_size])

    @property
    def total_size(self):
        return sum(val.reserved_size for val in self.data_arrays) + len(self.data_arrays)

    def write_offset(self, buffer_index: int, slot_index: int):
        start_of_array = sum(v.reserved_size for v in self.data_arrays[:buffer_index])

        shared_array = self.data_arrays[buffer_index]
        slot_size = int(np.prod(shared_array.data_shape) * shared_array.datatype_size)

        index_array = len(self.data_arrays)

        return index_array + start_of_array + slot_index * slot_size

    def clear(self):
        self._mm[:] = 0
        self._mm.flush()

    def __del__(self):
        """
        Destructor: Removing the bindings to the file (but not deleting the file!)
        """
        try:
            del self._mm
        except AttributeError:
            # In case the attributes are already deleted
            pass

    def init_file(self, filepath: Path):
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(b"\x00" * self.total_size)
                os.chmod(filepath, 0o666)
            return

        # Checking if size is correct
        with open(filepath, "ab") as f:  # append mode
            current_size = os.path.getsize(filepath)
            if self.total_size > current_size:
                f.write(b"\x00" * (self.total_size - current_size))

    # -----
    # write_index operations
    # -----
    def current_index(self, shared_array_index: int) -> int:
        return int(self._write_index_mm[shared_array_index])

    def set_write_index(self, shared_array_index: int, index_value: int):
        self._write_index_mm[shared_array_index] = index_value
        self._mm.flush()

    def increment_write_index(self, shared_array_index: int):
        next_index = self.current_index(shared_array_index) + 1
        if next_index == self.data_arrays[shared_array_index].reserved_count:
            next_index = 0  # This makes the write_index circular
        self.set_write_index(shared_array_index=shared_array_index, index_value=next_index)

    # -----
    # Write operations
    # -----
    def write_data(self, shared_array_index: int, input_data: np.ndarray):
        self.write_data_at(shared_array_index=shared_array_index,
                           write_index=self.current_index(shared_array_index),
                           input_data=input_data)
        self.increment_write_index(shared_array_index=shared_array_index)

    def write_data_at(self, shared_array_index: int, write_index: int, input_data: np.ndarray):
        shared_array = self.data_arrays[shared_array_index]
        array = np.frombuffer(input_data, dtype=np.uint8).ravel()
        start = self.write_offset(buffer_index=shared_array_index, slot_index=write_index)
        end = start + self.data_arrays[shared_array_index].slot_size
        self._mm[start:end] = array
        self._mm.flush()

    # -----
    # Read operations
    # -----
    def read_data_array(self, shared_array_index: int) -> np.ndarray:
        # Return a snapshot (copy) to avoid referencing live memory
        shared_array = self.data_arrays[shared_array_index]
        starting_pos = self.write_offset(buffer_index=shared_array_index, slot_index=0)

        shape = [shared_array.reserved_count, *shared_array.data_shape]
        return np.copy(np.ndarray(
            shape=shape,
            dtype=shared_array.datatype,
            buffer=self._mm[starting_pos:starting_pos + shared_array.reserved_size].view(
                shared_array.datatype)
        ))

    def read_data(self, shared_array_index: int, slot_index: int):
        shared_array = self.data_arrays[shared_array_index]
        start = self.write_offset(buffer_index=shared_array_index, slot_index=slot_index)
        buf = self._mm[start:start + shared_array.slot_size]
        return np.frombuffer(buf, dtype=shared_array.datatype).reshape(shared_array.data_shape).copy()


class CarlaWrapper:
    class CarlaDataType(IntEnum):
        images = 0
        object_detected = 1
        waypoint = 2
        lidar_points = 3
        object_tracking = 4

    def __init__(self, filename, image_width: int, image_height: int, max_lidar_points: int):
        data_arrays = [
            SharedMemoryArray(data_shape=[image_height, image_width, 3],  # Raw images
                              reserved_count=100,
                              datatype=np.uint8),
            SharedMemoryArray(data_shape=[image_height, image_width, 3],  # Object Detected images
                              reserved_count=100,
                              datatype=np.uint8),
            SharedMemoryArray(data_shape=[33],  # 33 for now
                              reserved_count=100,
                              datatype=np.float64),
            SharedMemoryArray(data_shape=[max_lidar_points, 4],  # LiDAR points x, y, z, intensity
                              reserved_count=100,
                              datatype=np.float32),
            SharedMemoryArray(data_shape=[image_height, image_width, 3],  # Object Tracking
                              reserved_count=100,
                              datatype=np.uint8),
        ]
        self.shared_memory = SharedMemoryManager(filename=filename,
                                                 data_arrays=data_arrays)

        self.max_lidar_points = max_lidar_points

    def clear(self):
        self.shared_memory.clear()

    # ----- Raw Images
    def write_image(self, image):
        # image.save_to_disk(f'/home/s0203301/project/images/{image.frame:08d}.png')

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # drop alpha
        array = np.ascontiguousarray(array)

        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.images.value, input_data=array)
        return

    def read_images(self) -> np.ndarray:
        return self.shared_memory.read_data_array(shared_array_index=self.CarlaDataType.images.value)

    def read_image(self, index: int) -> np.ndarray:
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=index)

    @property
    def latest_image_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.images.value)

    def read_latest_image(self) -> np.ndarray:
        slot_index = self.latest_image_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.images.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.images.value, slot_index=slot_index)

    # ----- Object Detected
    @property
    def object_detected_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.object_detected.value)

    def read_latest_object_detected(self) -> np.ndarray:
        slot_index = self.latest_image_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.object_detected.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.object_detected.value,
                                            slot_index=slot_index)

    def write_object_detected(self, image: np.ndarray):
        array = np.frombuffer(image, dtype=np.uint8)
        array = np.ascontiguousarray(array)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.object_detected.value, input_data=array)
        return

    # ----- Waypoints
    def write_waypoint(self, waypoint: np.ndarray):
        array = np.frombuffer(waypoint, dtype=np.float64)
        array = np.ascontiguousarray(array)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.waypoint.value, input_data=array)
        return

    def read_waypoints(self) -> np.ndarray:
        return self.shared_memory.read_data_array(shared_array_index=self.CarlaDataType.waypoint.value)

    # ----- LiDAR
    @property
    def latest_lidar_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.lidar_points.value)

    def write_lidar_points(self, points: np.ndarray):
        """
        Write a LiDAR frame to shared memory.
        Points shape: (N,4) with x,y,z,intensity
        Will truncate if N > MAX_LIDAR_POINTS
        """
        points = np.ascontiguousarray(points[:self.max_lidar_points], dtype=np.float32)

        # Pad with zeros if less points than max
        if points.shape[0] < self.max_lidar_points:
            pad = np.zeros((self.max_lidar_points - points.shape[0], 4), dtype=np.float32)
            points = np.vstack((points, pad))

        self.shared_memory.write_data(
            shared_array_index=self.CarlaDataType.lidar_points.value,
            input_data=points
        )

    def read_latest_lidar_points(self) -> np.ndarray:
        slot_index = self.latest_lidar_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.lidar_points.value].reserved_count - 1
        points = self.shared_memory.read_data(
            shared_array_index=self.CarlaDataType.lidar_points.value,
            slot_index=slot_index
        )
        return points

    # ----- Object Tracking
    @property
    def latest_object_tracking_index(self) -> int:
        return self.shared_memory.current_index(shared_array_index=self.CarlaDataType.object_tracking.value)

    def read_latest_object_tracking(self) -> np.ndarray:
        slot_index = self.latest_object_tracking_index - 1
        if slot_index == -1:
            slot_index = self.shared_memory.data_arrays[self.CarlaDataType.object_tracking.value].reserved_count - 1
        return self.shared_memory.read_data(shared_array_index=self.CarlaDataType.object_tracking.value,
                                            slot_index=slot_index)

    def write_object_tracking(self, image: np.ndarray):
        array = np.frombuffer(image, dtype=np.uint8)
        array = np.ascontiguousarray(array)
        self.shared_memory.write_data(shared_array_index=self.CarlaDataType.object_tracking.value, input_data=array)
        return


# ==============================================================================
# -- Point Accumulator ---------------------------------------------------------
# ==============================================================================
class PointAccumulator:
    """
    Accumulates LiDAR packets to create a dense point cloud similar to KITTI.
    """

    def __init__(self, target_points=60000, max_buffers=10):
        self.buffer: Deque[np.ndarray] = deque(maxlen=max_buffers)
        self.target_points = target_points

    def add_points(self, points: np.ndarray):
        # Filter out padding (0,0,0)
        mask = np.any(points[:, :3] != 0, axis=1)
        valid_points = points[mask]

        if len(valid_points) > 0:
            self.buffer.append(valid_points)

    def get_sweep(self) -> np.ndarray:
        # Stack all buffered frames
        return np.concatenate(list(self.buffer), axis=0)

    def is_ready(self) -> bool:
        # Check if we have enough density
        if not self.buffer: return False
        total = sum(len(b) for b in self.buffer)
        return total >= self.target_points

# Add the repo to python path so we can import its modules
sys.path.append("/workspace/PointPillars")

from pointpillars.model import PointPillars
from pointpillars.utils import read_points, keep_bbox_from_lidar_range

class PointPillarsML:
    def __init__(self, ckpt_path):
        self.logger = logging.getLogger("PointPillarsML")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initializing model on {self.device}...")

        # 1. Define Model Settings (Must match what the model was trained with!)
        # These are standard KITTI settings used in zhulf0804's repo
        self.CLASSES = {
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2
        }
        self.pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.voxel_size = [0.16, 0.16, 4]

        # 2. Initialize the Network
        self.model = PointPillars(nclasses=len(self.CLASSES))

        # 3. Load Weights
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if not checkpoint:
            raise InvalidArgumentError(f"Could not open checkpoint at {ckpt_path}")
        self.logger.info(f"Loaded checkpoint from {ckpt_path}")

        self.model.load_state_dict(checkpoint)
        self.logger.info("Model loaded successfully.")

        self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device}.")

        self.model.eval()
        self.logger.info("Set to eval and ready to run")

    @classmethod
    def point_range_filter(cls,
                           pts,
                           point_range: Optional[list] = None):
        """
        Filter points that are outside the range required by PointPillars
        """
        point_range = point_range or [0, -39.68, -3, 69.12, 39.68, 1]

        flag_x_low = pts[:, 0] > point_range[0]
        flag_y_low = pts[:, 1] > point_range[1]
        flag_z_low = pts[:, 2] > point_range[2]
        flag_x_high = pts[:, 0] < point_range[3]
        flag_y_high = pts[:, 1] < point_range[4]
        flag_z_high = pts[:, 2] < point_range[5]
        keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
        return pts[keep_mask]

    def preprocess(self, numpy_points):
        """
        Converts CARLA points (N, 4) -> PointPillars Tensor
        """
        points = numpy_points.copy()

        # 1. Filter empty padding points (0,0,0,0) from shared memory
        mask = np.any(points[:, :3] != 0, axis=1)
        points = points[mask]

        if len(points) == 0:
            return None

        # 2. Coordinate Transform (CARLA -> KITTI)
        # CARLA: x=front, y=right, z=up
        # KITTI: x=front, y=left, z=up
        points[:, 1] = -points[:, 1]

        # 3. Apply Point Range Filter (Crucial step from your example!)
        # This removes points that are too far away or behind the car
        points = self.point_range_filter(points)

        if len(points) == 0:
            return None

        # 4. Convert to Tensor and Move to GPU
        points_tensor = torch.from_numpy(points).float()
        return points_tensor.to(self.device)

    def predict(self, points_tensors: List[Tensor]):
        """
        The model expects a LIST of tensors (one per frame in the batch)
        Runs the actual inference
        """
        with torch.no_grad():
            # Run model
            results = self.model(batched_pts=points_tensors, mode='test')

            # results is a dict:
            # {
            #   'lidar_bboxes': Tensor(N, 7),
            #   'labels': Tensor(N),
            #   'scores': Tensor(N)
            # }
            return results

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ml_engine = PointPillarsML(ckpt_path="/workspace/PointPillars/pretrained/epoch_160.pth") # This could be env variable
    logger.info(f"${ml_engine.__class__} class instantiated")

    camera_width = 800
    camera_height = 600
    max_lidar_points = 120000
    shared_memory_filepath = "/dev/shm/carla_shared/carla_shared_v6.dat"
    shared_memory = CarlaWrapper(filename=shared_memory_filepath,
                                 image_width=camera_width,
                                 image_height=camera_height,
                                 max_lidar_points=max_lidar_points)
    logger.info("Shared memory setup")

    # Pyro server
    PYRO_URI = "PYRO:pyrostateserver@localhost:9090"
    remote_monitor = Pyro4.Proxy(PYRO_URI)
    logger.info(f"Connected to Pyro Server at {PYRO_URI}")

    # Data loader
    accumulator = PointAccumulator(target_points=40000, max_buffers=3)

    logger.info("Starting loop")
    while True:
        point_cloud: np.ndarray = shared_memory.read_latest_lidar_points() # 1x4 np.array, x, y, z and intensity
        accumulator.add_points(point_cloud)

        # Check if we have enough density
        if not accumulator.is_ready():
            continue

        dense_cloud = accumulator.get_sweep()
        tensor_input = ml_engine.preprocess(dense_cloud)
        if tensor_input is None:
            logger.info("No valid LiDAR points after preprocessing. Skipping frame.")
            continue

        # todo do batching instead of only one frame (like 5->10 maybe?)
        result = ml_engine.predict([tensor_input])

        # fixme return object handling -> might not be required to do the instance check but idk how the library works
        if result:
            # When doing the batching we get multiple returns, we want the most recent one
            if isinstance(result, list):
                result = result[-1]

            #
            if isinstance(result, dict):
                bboxes = result['lidar_bboxes']  # [x, y, z, w, l, h, rot]
                labels = result['labels']  # [0, 1, 2] -> Car, Ped, Cyc
                scores = result['scores']  # 0.0 to 1.0
                raw_detects = len(bboxes)

                mask = scores > 0.45
                bboxes = bboxes[mask].tolist()
                labels = labels[mask].tolist()
                scores = scores[mask].tolist()
            else:
                bboxes = None
                labels = None
                scores = None
                raw_detects = None

            try:
                remote_monitor.update_lidar_result(
                    bboxes,
                    labels,
                    scores,
                    raw_detects
                )
            except Exception as e:
                logger.warning(e)

            logger.info(f"Found ${len(bboxes or [])} bboxes")
        else:
            logger.info("No bboxes found")