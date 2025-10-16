# #!/usr/bin/env python3
# # carla_reader_dat.py
# # Read positions + latest RGB frame from carla_shared.dat
#
# import argparse
# import sys
# from pathlib import Path
# import numpy as np
#
# try:
#     from PIL import Image  # pip install pillow
# except Exception:
#     Image = None
#
# # --- MUST MATCH WRITER ---
# MAX_ACTORS = 100
# IMAGE_WIDTH = 320
# IMAGE_HEIGHT = 240
# IMAGE_CHANNELS = 4  # writer stores RGB
#
# pos_shape = (MAX_ACTORS, 3)                     # float64 (x,y,z)
# img_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
#
# POS_BYTES = (MAX_ACTORS * 3) * 8                # float64 -> 8 bytes each
# IMG_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
# TOTAL_BYTES = POS_BYTES + IMG_BYTES
#
#
# def open_memmaps(filename: Path):
#     if not filename.exists():
#         sys.exit(f"[ERR] File not found: {filename}")
#
#     size = filename.stat().st_size
#     if size != TOTAL_BYTES:
#         print(f"[WARN] File size is {size} bytes but expected {TOTAL_BYTES}. "
#               f"Check MAX_ACTORS / image dims in writer & reader match.")
#
#     pos_mm = np.memmap(filename, dtype=np.float64, mode="r",
#                        shape=pos_shape, offset=0)
#     img_mm = np.memmap(filename, dtype=np.uint8, mode="r",
#                        shape=img_shape, offset=POS_BYTES)
#     return pos_mm, img_mm
#
#
# def save_image_from_array(img_arr: np.ndarray, out_path: Path):
#     if Image is None:
#         sys.exit("[ERR] Pillow not installed. Run: pip install pillow")
#     # Ensure it's HxWx3 uint8
#     arr = np.asarray(img_arr, dtype=np.uint8)
#     if arr.shape != img_shape:
#         sys.exit(f"[ERR] Unexpected image shape {arr.shape}, expected {img_shape}")
#     Image.fromarray(arr, mode="RGB").save(out_path)
#     print(f"[OK] Wrote {out_path}")
#
#
# def read_file(filename: Path):
#     """
#     Open the shared .dat file, print a short summary, and return (pos_mm, img_mm).
#     NOTE: pos_mm and img_mm are numpy.memmap views (zero-copy). If you need a
#     snapshot that won't change under you, wrap with np.array(...).
#     """
#     pos_mm, img_mm = open_memmaps(filename)
#
#     # --- Print positions (first 10) ---
#     print("=== Actor positions (first 10) ===")
#     # snapshot to avoid printing changing data mid-read
#     print(np.array(pos_mm[:10]))
#
#     # --- Print image stats ---
#     img = np.array(img_mm)  # take a snapshot of the current frame
#     print("\n=== Image info ===")
#     print(f"shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}, mean={img.mean():.2f}")
#
#     # Tiny 3x3 patch from top-left (RGB triples) for sanity
#     patch = img[:3, :3, :].reshape(-1, 3)
#     print("\nTop-left 3x3 RGB patch:")
#     print(patch)
#
#     return pos_mm, img_mm
#
# def image_show(pos_mm: np.ndarray, img_mm: np.ndarray):
#     if img.shape[2] == 4:
#         rgb = img[:, :, :3][:, :, ::-1]  # drop A, BGR->RGB
#     else:
#         rgb = img
#
#     plt.imshow(rgb)
#     plt.axis('off')
#     plt.title("CARLA latest frame")
#     plt.show()
#
# def main():
#     ap = argparse.ArgumentParser(description="Read CARLA shared .dat (poses + latest RGB frame)")
#     ap.add_argument("--file", "-f", type=Path, default=Path("/dev/shm/carla_shared.dat"),
#                     help="Path to .dat file (default: /dev/shm/carla_shared.dat)")
#     ap.add_argument("--out", "-o", type=Path, default=Path("latest_frame.png"),
#                     help="Where to save the PNG (default: latest_frame.png)")
#     ap.add_argument("--show-poses", action="store_true",
#                     help="Print first 10 actor positions")
#     ap.add_argument("--watch", action="store_true",
#                     help="Continuously read & overwrite the PNG (Ctrl+C to stop)")
#     ap.add_argument("--interval", type=float, default=0.1,
#                     help="Seconds between reads in --watch mode (default: 0.1s)")
#     args = ap.parse_args()
#     path = Path("/dev/shm/carla_shared.dat")
#     pos_mm, img_mm = open_memmaps(args.file)
#
#     def one_shot():
#         if args.show_poses:
#             print("First 10 actor positions (x, y, z):")
#             print(np.array(pos_mm[:10]))
#         # copy from shared buffer and save
#         img = np.array(img_mm)  # take a snapshot
#         print("hallo?")
#         read_file(args.file)
#         save_image_from_array(img, args.out)
#
#         # print("pos: "+str(pos_mm[:10]))
#         # print("img: "+str(img_mm[:10]))
#
#     if args.watch:
#         import time
#         print(f"[INFO] Watching {args.file} → {args.out} (Ctrl+C to stop)")
#         try:
#             while True:
#                 one_shot()
#                 time.sleep(args.interval)
#         except KeyboardInterrupt:
#             print("\n[INFO] Stopped.")
#     else:
#         one_shot()
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# carla_reader_dat.py

import argparse, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image  # pip install pillow
except Exception:
    Image = None

# --- MUST MATCH WRITER ---
MAX_ACTORS = 100
IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = 320, 240, 4  # writer currently stores 4ch (BGRA)
pos_shape = (MAX_ACTORS, 3)
img_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
POS_BYTES = (MAX_ACTORS * 3) * 8
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

def to_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 RGB uint8. If 4ch, assume BGRA from CARLA and convert."""
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {img.shape}")
    if img.shape[2] == 3:
        return img.astype(np.uint8, copy=False)
    if img.shape[2] == 4:
        # BGRA -> BGR, drop A, then BGR -> RGB
        bgr = img[:, :, :3]
        rgb = bgr[:, :, ::-1]
        return rgb.astype(np.uint8, copy=False)
    raise ValueError(f"Unsupported channel count: {img.shape[2]}")

def save_image_from_array(img_arr: np.ndarray, out_path: Path):
    if Image is None:
        sys.exit("[ERR] Pillow not installed. Run: pip install pillow")
    rgb = to_rgb(img_arr)
    Image.fromarray(rgb, mode="RGB").save(out_path)
    print(f"[OK] Wrote {out_path}")

def read_file(filename: Path):
    """Print a short summary and return (pos_mm, img_mm, img_snapshot)."""
    pos_mm, img_mm = open_memmaps(filename)

    print("=== Actor positions (first 10) ===", flush=True)
    print(np.array(pos_mm[:10]), flush=True)

    img = np.array(img_mm)  # snapshot
    print("\n=== Image info ===", flush=True)
    print(f"shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.2f}", flush=True)

    patch = img[:3, :3, :min(3, img.shape[2])].reshape(-1, min(3, img.shape[2]))
    print("\nTop-left 3x3 patch (first 3 channels):", flush=True)
    print(patch, flush=True)

    return pos_mm, img_mm, img

def show_image(img: np.ndarray, title: str = "CARLA latest frame"):
    # Lazy import so matplotlib is optional
    import matplotlib.pyplot as plt
    rgb = to_rgb(img)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Read CARLA shared .dat (poses + latest image)")
    ap.add_argument("--file", "-f", type=Path, default=Path("/dev/shm/carla_shared.dat"),
                    help="Path to .dat file (default: /dev/shm/carla_shared.dat)")
    ap.add_argument("--out", "-o", type=Path, default=Path("latest_frame.png"),
                    help="Where to save the PNG")
    ap.add_argument("--watch", action="store_true",
                    help="Continuously read, display and save (Ctrl+C to stop)")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="Seconds between reads in --watch mode")
    args = ap.parse_args()

    def one_shot():
        _, _, img = read_file(args.file)   # prints happen here
        save_image_from_array(img, args.out)
        try:
            show_image(img)
        except ModuleNotFoundError:
            print("[INFO] matplotlib not installed; skipping on-screen display. "
                  "Install with: python3.12 -m pip install matplotlib")

    if args.watch:
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

