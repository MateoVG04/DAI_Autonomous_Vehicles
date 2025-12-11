import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import time
import os

# --- Imports ---
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
from model import UNet
# We reuse the mathematical logic class you tuned previously
from Machine_Vision.lane_post_processing import FastLaneDetector

# =========================
# Config
# =========================
# --- Model ---
MODEL_PATH = "/home/shared/3_12_jupyter/bin/simulation//Model/unet_multiclass.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Shared Memory ---
camera_width = 800
camera_height = 600
shared_memory_filepath = "/dev/shm/carla_shared_Rune.dat"

# --- Initialize Math Processor ---
# This uses the logic we tuned (Horizon 320, Top Width 60)
detector = FastLaneDetector(img_h=camera_height, img_w=camera_width)

# =========================
# 1. Load U-Net Model (Multi-Class)
# =========================
print(f"Loading Multi-Class U-Net model from: {MODEL_PATH}")
model = UNet(n_channels=3, n_classes=3) # 3 Classes!
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on device: {DEVICE}")

# =========================
# 2. Preprocessing
# =========================
preprocess_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 3. Shared Memory
# =========================
shared_memory = CarlaWrapper(
    filename=shared_memory_filepath,
    image_width=camera_width,
    image_height=camera_height
)

# =========================
# 4. Helper Functions
# =========================
def predict_multiclass(bgr_image):
    """
    Returns a mask of IDs (0, 1, 2).
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    input_tensor = preprocess_transform(image_pil)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)

    # Argmax to get class IDs (0, 1, 2)
    _, predicted = torch.max(output, 1)
    mask_ids = predicted.squeeze().cpu().numpy().astype(np.uint8)

    return mask_ids

# =========================
# 5. Main Processing Loop
# =========================
if __name__ == "__main__":
    print("--- Starting Advanced Lane Detection ---")
    print("Press 'q' to quit.")

    time.sleep(1)

    while True:
        try:
            # 1. Read Frame
            frame_bgr = shared_memory.read_latest_image()

            if frame_bgr is None or frame_bgr.size == 0:
                time.sleep(0.1)
                continue

            # 1. Get Multi-Class Prediction (256x512)
            mask_ids_small = predict_multiclass(frame_bgr)

            # 2. Resize to Full Resolution (800x600)
            # Important: Use INTER_NEAREST to keep IDs as 0, 1, 2
            mask_ids_full = cv2.resize(mask_ids_small, (camera_width, camera_height), interpolation=cv2.INTER_NEAREST)

            # 3. Process
            # Just pass the raw IDs. The detector handles the logic.
            final_image = detector.process(mask_ids_full, frame_bgr)


            try:
                cv2.imshow("Advanced Lane Detection", final_image)
                if cv2.waitKey(1) == ord('q'):
                    print("'q' pressed.")
                    break
            except Exception:
                pass # Window failed (Headless), but file saved

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()