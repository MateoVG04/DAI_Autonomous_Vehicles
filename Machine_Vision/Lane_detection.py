import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import time
import os

# --- Imports ---
from Machine_Vision.lane_post_processing import FastLaneDetector
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
from model import UNet

# =========================
# Config
# =========================
# --- Model and Device ---
MODEL_PATH = "/home/shared/3_12_jupyter/bin/simulation//Model/unet_lanes_not_lights.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Shared Memory and Camera ---
camera_width = 800
camera_height = 600
shared_memory_filepath = "/dev/shm/carla_shared_Rune.dat"

# --- Initialize PolyFitter ---
# Initialize with full resolution for correct polygon drawing
detector = FastLaneDetector(img_h=600, img_w=800)
# =========================
# 1. Load U-Net Model
# =========================
print(f"Loading U-Net model from: {MODEL_PATH}")
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded successfully on device: {DEVICE}")

# =========================
# 2. Define Preprocessing Transforms
# =========================
preprocess_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 3. Initialize Shared Memory
# =========================
shared_memory = CarlaWrapper(
    filename=shared_memory_filepath,
    image_width=camera_width,
    image_height=camera_height
)


# =========================
# 4. Helper Functions
# =========================
def predict_lanes(bgr_image):
    """
    Takes a BGR NumPy image, preprocesses it, and returns a binary lane mask.
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb_image)
    input_tensor = preprocess_transform(image_pil)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.sigmoid(output[0])
    mask = probs.cpu().numpy()

    mask = np.squeeze(mask)
    # Threshold: Values > 0.8 become 255 (White), others 0 (Black)
    final_mask = (mask > 0.8).astype(np.uint8) * 255

    return final_mask


# =========================
# 5. Main Processing Loop
# =========================
if __name__ == "__main__":
    print("--- Starting Lane Detection (Polynomial Mode) ---")
    print("Press 'q' in the window to quit.")

    time.sleep(1)
    frame_counter = 0

    while True:
        try:
            # --- 1) Read the latest raw frame ---
            frame_bgr = shared_memory.read_latest_image()

            if frame_bgr is None or frame_bgr.size == 0:
                time.sleep(0.1)
                continue

                # ... Inside the loop ...

            # 1. Get raw mask from U-Net (256x512)
            raw_mask = predict_lanes(frame_bgr)

            # 2. Resize mask to full resolution (800x600)
            # The logic works best at full res
            mask_full = cv2.resize(raw_mask, (800, 600), interpolation=cv2.INTER_NEAREST)

            # 3. Process and Draw
            # Pass the mask AND the original frame. It handles everything.
            final_image = detector.process(mask_full, frame_bgr)


            # --- 6) Display Only (No Save) ---
            try:
                cv2.imshow("Lane Detection", final_image)
                if cv2.waitKey(1) == ord('q'):
                    print("'q' key pressed. Exiting.")
                    break
            except Exception as e:
                # If display fails (e.g. no X11 forwarding), just print status
                pass


        except KeyboardInterrupt:
            print("Keyboard interrupt. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred in the loop: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()
    print("\n--- Viewer Closed ---")