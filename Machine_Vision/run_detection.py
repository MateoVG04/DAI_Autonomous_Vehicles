import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import time
import os

from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
from model import UNet

# IMPORT TRACKER
from Machine_Vision.lane_post_processing import DirectLaneTracer

# =========================
# Config
# =========================
MODEL_PATH = "/home/shared/3_12_jupyter/bin/simulation/Model/unet_multiclass.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

camera_width = 800
camera_height = 600

shm_camera_path = "/dev/shm/carla_shared_Rune.dat"
shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"

# Initialize Tracker
tracker = DirectLaneTracer(img_h=camera_height, img_w=camera_width)

# =========================
# 1. Load U-Net
# =========================
print(f"Loading Model from: {MODEL_PATH}")
model = UNet(n_channels=3, n_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded on: {DEVICE}")

# =========================
# 2. Setup
# =========================
preprocess_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cam_shm = CarlaWrapper(shm_camera_path, camera_width, camera_height, 100000)
lid_shm = CarlaWrapper(shm_lidar_path, camera_width, camera_height, 100000)

def predict_multiclass(bgr_image):
    # Convert BGR (OpenCV) to RGB (PyTorch)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    input_tensor = preprocess_transform(image_pil)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)

    _, predicted = torch.max(output, 1)
    mask_ids = predicted.squeeze().cpu().numpy().astype(np.uint8)
    return mask_ids

# =========================
# 4. Main Loop
# =========================
if __name__ == "__main__":
    print("--- Starting Direct Lane & Lidar Debug ---")
    print("Press 'q' to quit.")

    time.sleep(1)

    while True:
        try:
            raw_image = cam_shm.read_latest_image()
            lidar_bev = lid_shm.read_latest_object_detected()

            if raw_image is None or raw_image.size == 0:
                time.sleep(0.01)
                continue

            # --- COLOR FIX ---
            # You confirmed mode [1, 2, 0] worked.
            # This logic assumes raw_image is [G, B, R] or similar weird mapping.
            # We enforce this swap to get BGR.
            frame_bgr = raw_image[:, :, [1, 2, 0]].copy()

            # Predict
            mask_ids_small = predict_multiclass(frame_bgr)
            mask_ids_full = cv2.resize(mask_ids_small, (camera_width, camera_height), interpolation=cv2.INTER_NEAREST)

            # Process
            dashboard = tracker.process(mask_ids_full, lidar_bev, frame_bgr)

            try:
                cv2.imshow("Lane Detection Debug Dashboard", dashboard)
                if cv2.waitKey(1) == ord('q'):
                    break
            except Exception:
                pass

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

    cv2.destroyAllWindows()