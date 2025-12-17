import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Import your tracker
from Machine_Vision.lane_post_processing import VisualLaneTracer
from Machine_Vision.model import UNet


class DistanceSystem:
    def __init__(self, model_path, width=800, height=600, fov=90.0, device='cuda'):
        self.w = width
        self.h = height

        # Robust device selection
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. LOAD MODEL
        print(f"Loading Perception Model from {model_path}...")
        self.model = UNet(n_channels=3, n_classes=3)
        # Handle map_location for CPU/GPU fallback
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 2. SETUP TRANSFORMS
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. SETUP TRACKER
        self.tracker = VisualLaneTracer(img_h=height, img_w=width)

        # 4. CALIBRATION (Intrinsics)
        focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.identity(3)
        self.K[0, 0] = self.K[1, 1] = focal
        self.K[0, 2] = width / 2.0
        self.K[1, 2] = height / 2.0

        self.lidar_range = 50.0

    def compute(self, rgb_image, lidar_data):
        """
        rgb_image: BGR Image (H, W, 3)
        lidar_data: Raw CARLA sensor data OR Numpy Array
        """
        # A. PREPARE RGB
        if rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]
        img_for_net = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # B. RUN U-NET
        mask = self._predict_mask(img_for_net)

        # C. PROCESS LIDAR
        # Handle both Raw Bytes (Direct) and Numpy Array (Shared Memory)
        depth_img = self._process_lidar(lidar_data)

        # D. RUN LANE TRACER
        mask_full = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        dashboard = self.tracker.process(mask_full, depth_img, rgb_image)

        return self.tracker.current_distance, dashboard

    def _predict_mask(self, rgb_image):
        image_pil = Image.fromarray(rgb_image)
        input_tensor = self.preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)

        _, predicted = torch.max(output, 1)
        return predicted.squeeze().cpu().numpy().astype(np.uint8)

    def _process_lidar(self, packed_data):
        if packed_data is None: return None

        # --- FIX: Handle different input types ---
        if hasattr(packed_data, 'ravel'):
            # It's a NumPy array (from Shared Memory fake image)
            raw_bytes = packed_data.ravel()
        else:
            # It's bytes or memoryview (Direct from CARLA)
            # Use as-is, no need to ravel
            raw_bytes = packed_data

        # Convert to float32 points
        try:
            # Calculate valid bytes (multiple of 16)
            # raw_bytes might be a memoryview, len() works on it
            data_len = len(raw_bytes)
            valid_bytes = data_len - (data_len % 16)

            # np.frombuffer works with both bytes and memoryview
            points = np.frombuffer(raw_bytes[:valid_bytes], dtype=np.dtype('f4'))
            points = np.reshape(points, (int(len(points) / 4), 4))
        except Exception as e:
            # print(f"Lidar Decode Error: {e}")
            return None

        # 2. FILTER
        lidar_x = points[:, 0]
        lidar_y = points[:, 1]
        lidar_z = points[:, 2]

        # Filter: Front only, Max Range, Ignore Road
        mask = (lidar_x > 1.0) & (lidar_x < self.lidar_range) & (lidar_z > -1.5)

        lidar_x = lidar_x[mask]
        lidar_y = lidar_y[mask]
        lidar_z = lidar_z[mask]

        if len(lidar_x) == 0: return None

        # 3. PROJECT (3D -> 2D)
        u_coords = (lidar_y * self.K[0, 0] / lidar_x) + self.K[0, 2]
        v_coords = (-lidar_z * self.K[1, 1] / lidar_x) + self.K[1, 2]

        # Screen Bounds
        screen_mask = (u_coords >= 0) & (u_coords < self.w) & \
                      (v_coords >= 0) & (v_coords < self.h)

        u_coords = u_coords[screen_mask].astype(int)
        v_coords = v_coords[screen_mask].astype(int)
        depths = lidar_x[screen_mask]

        # 4. DRAW DEPTH MAP
        depth_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Normalize depth
        norm_depth = np.clip(1.0 - (depths / self.lidar_range), 0, 1)
        intensity = (norm_depth * 255).astype(np.uint8)

        for i in range(len(u_coords)):
            val = int(intensity[i])
            cv2.circle(depth_img, (u_coords[i], v_coords[i]), 2, (val, val, val), -1)

        return depth_img