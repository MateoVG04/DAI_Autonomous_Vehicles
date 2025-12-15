from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from simulation.python_3_8_20_scripts.shared_memory_utils import CarlaWrapper
import time

# =========================
# Config
# =========================
MODEL_PATH = "yolo11n.pt"
CONF = 0.4

camera_width = 800
camera_height = 600
shared_memory_filepath = "/dev/shm/carla_shared.dat"

# Toon figuren? (True = laat origineel + detectie in 1 figure zien)
SHOW_PREVIEW = True

# =========================
# Init model + shared memory
# =========================
model = YOLO(MODEL_PATH)
shared_memory = CarlaWrapper(
    filename=shared_memory_filepath,
    image_width=camera_width,
    image_height=camera_height
)

def process_once():
    """
    1) Lees laatste ruw frame (BGR, HxWx3)
    2) YOLO detectie (RGB input)
    3) Annotatie (BGR) -> terugschrijven naar object_detected
    4) Verificatie: lees laatste object_detected terug en toon samen met origineel
    """
    # --- 1) Lees laatste ruwe frame (BGR, HxWx3) ---
    frame_bgr = shared_memory.read_latest_image()  # jouw wrapper corrigeert index al
    if frame_bgr is None or frame_bgr.size == 0:
        print("Geen geldig frame gevonden in buffer.")
        return

    # --- 2) YOLO verwacht RGB ---
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # --- 3) YOLO detectie ---
    results = model(source=frame_rgb, conf=CONF, save=False, show=False)
    result = results[0]

    # --- 4) Geannoteerd beeld (BGR) ---
    annotated_bgr = result.plot()  # shape (H, W, 3), dtype=uint8

    # --- 5) Schrijf geannoteerde afbeelding terug naar object_detected-buffer ---
    shared_memory.write_object_detected(annotated_bgr)

    # --- 6) Verificatie: lees laatste object_detected terug ---
    det_bgr = shared_memory.read_latest_object_detected()  # (H, W, 3) BGR

    # --- 7) (Optioneel) Toon origineel vs detectie ---
    # if SHOW_PREVIEW:
    #     plt.figure(figsize=(12, 5))
    #
    #     # Origineel
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    #     plt.title("Origineel (laatste frame)")
    #     plt.axis("off")
    #
    #     # Detectie
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB))
    #     plt.title("YOLO11 detectie (verificatie uit buffer)")
    #     plt.axis("off")
    #
    #     plt.tight_layout()
    #     plt.show(block=False)
    #     plt.pause(0.001)
        # plt.close()  # sluit desgewenst automatisch

    print("Detectie uitgevoerd, geannoteerd beeld wegschreven en geverifieerd.")

if __name__ == "__main__":
    while True:
        process_once()
