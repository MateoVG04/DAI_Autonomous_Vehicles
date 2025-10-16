#!/usr/bin/env python3
# carla_reader_dat.py
# Read positions + latest RGB frame from carla_shared.dat

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image  # pip install pillow
except Exception:
    Image = None

# --- MUST MATCH WRITER ---
MAX_ACTORS = 100
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3  # writer stores RGB

pos_shape = (MAX_ACTORS, 3)                     # float64 (x,y,z)
img_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

POS_BYTES = (MAX_ACTORS * 3) * 8                # float64 -> 8 bytes each
IMG_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
TOTAL_BYTES = POS_BYTES + IMG_BYTES


def open_memmaps(filename: Path):
    if not filename.exists():
        sys.exit(f"[ERR] File not found: {filename}")

    size = filename.stat().st_size
    if size != TOTAL_BYTES:
        print(f"[WARN] File size is {size} bytes but expected {TOTAL_BYTES}. "
              f"Check MAX_ACTORS / image dims in writer & reader match.")

    pos_mm = np.memmap(filename, dtype=np.float64, mode="r",
                       shape=pos_shape, offset=0)
    img_mm = np.memmap(filename, dtype=np.uint8, mode="r",
                       shape=img_shape, offset=POS_BYTES)
    return pos_mm, img_mm


def save_image_from_array(img_arr: np.ndarray, out_path: Path):
    if Image is None:
        sys.exit("[ERR] Pillow not installed. Run: pip install pillow")
    # Ensure it's HxWx3 uint8
    arr = np.asarray(img_arr, dtype=np.uint8)
    if arr.shape != img_shape:
        sys.exit(f"[ERR] Unexpected image shape {arr.shape}, expected {img_shape}")
    Image.fromarray(arr, mode="RGB").save(out_path)
    print(f"[OK] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Read CARLA shared .dat (poses + latest RGB frame)")
    ap.add_argument("--file", "-f", type=Path, default=Path("carla_shared.dat"),
                    help="Path to .dat file (default: carla_shared.dat)")
    ap.add_argument("--out", "-o", type=Path, default=Path("latest_frame.png"),
                    help="Where to save the PNG (default: latest_frame.png)")
    ap.add_argument("--show-poses", action="store_true",
                    help="Print first 10 actor positions")
    ap.add_argument("--watch", action="store_true",
                    help="Continuously read & overwrite the PNG (Ctrl+C to stop)")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="Seconds between reads in --watch mode (default: 0.1s)")
    args = ap.parse_args()

    pos_mm, img_mm = open_memmaps(args.file)

    def one_shot():
        if args.show_poses:
            print("First 10 actor positions (x, y, z):")
            print(np.array(pos_mm[:10]))
        # copy from shared buffer and save
        img = np.array(img_mm)  # take a snapshot
        save_image_from_array(img, args.out)

    if args.watch:
        import time
        print(f"[INFO] Watching {args.file} → {args.out} (Ctrl+C to stop)")
        try:
            while True:
                one_shot()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
    else:
        one_shot()


if __name__ == "__main__":
    main()
