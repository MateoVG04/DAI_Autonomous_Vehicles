import cv2
import time
import numpy as np
import os
from enum import IntEnum
from pathlib import Path
from typing import List


# ==============================================================================
# PASTE YOUR 'SharedMemoryArray', 'SharedMemoryManager', 'CarlaWrapper' CLASSES HERE
# ==============================================================================
# [ ... Paste your code from the prompt here ... ]
# ==============================================================================

def main():
    print("Initializing Viewer...")

    width = 800
    height = 600

    # Point to the Lidar file
    shm_lidar_path = "/dev/shm/carla_shared_Rune_Lidar.dat"

    # Initialize the wrapper in read mode
    # (The class handles opening existing files automatically)
    if not os.path.exists(shm_lidar_path):
        print(f"Waiting for file: {shm_lidar_path}")
        while not os.path.exists(shm_lidar_path):
            time.sleep(1)

    lidar_wrapper = CarlaWrapper(filename=shm_lidar_path, image_width=width, image_height=height)

    print("Viewer Running. Press 'q' to exit.")

    while True:
        try:
            # We used 'write_object_detected' in the simulation,
            # so we use 'read_latest_object_detected' here.
            lidar_img = lidar_wrapper.read_latest_object_detected()

            # Ensure it's contiguous for OpenCV
            lidar_img = np.ascontiguousarray(lidar_img)

            # Draw a marker for the Ego Vehicle
            center_x, center_y = width // 2, height // 2
            cv2.arrowedLine(lidar_img, (center_x, center_y + 10), (center_x, center_y - 10), (0, 0, 255), 2)

            # OpenCV usually prefers BGR, but your Lidar colors are manually set,
            # so RGB vs BGR just swaps Green/Blue channels.
            # Since we used Green for ground and Blue for height, it might look Red/Green in BGR.
            # Optional: cv2.cvtColor(lidar_img, cv2.COLOR_RGB2BGR)

            cv2.imshow("Lidar BEV", lidar_img)

        except Exception as e:
            # Often happens if read occurs while write is flushing (rare with your setup but possible)
            # or if the file isn't populated yet
            # print(f"Read error: {e}")
            pass

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()