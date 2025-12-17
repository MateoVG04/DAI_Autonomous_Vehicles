import os
import random

# --- CONFIGURATION ---
# Source: Where the actual Label files are
DATA_ROOT = "/workspace/carla_kitti"

# Destination: Where PointPillars code expects the list files
OUTPUT_DIR = "/workspace/PointPillars/pointpillars/dataset/ImageSets"


# ---------------------

def create_splits(train_ratio=0.75):
    label_dir = os.path.join(DATA_ROOT, "training", "label_2")
    image_dir = os.path.join(DATA_ROOT, "training", "image_2")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Scanning labels in: {label_dir}")
    print(f"Checking images in: {image_dir}")

    if not os.path.exists(label_dir):
        print(f"CRITICAL ERROR: Label directory not found at {label_dir}")
        return

    valid_ids = []
    skipped_no_car = 0
    skipped_no_image = 0
    total_files = 0

    # 1. Collect all valid frames (Strict Filter)
    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue

        total_files += 1
        frame_id = os.path.splitext(filename)[0]
        filepath = os.path.join(label_dir, filename)

        # --- CHECK 1: DOES IMAGE EXIST? ---
        image_path = os.path.join(image_dir, f"{frame_id}.png")
        if not os.path.exists(image_path):
            skipped_no_image += 1
            # Optional: Print warning for first few missing images
            if skipped_no_image <= 5:
                print(f"Warning: Missing image for label {frame_id}")
            continue

        # --- CHECK 2: DOES LABEL HAVE A CAR? ---
        has_car = False
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(' ')
                    if len(parts) > 0:
                        if parts[0] == 'Car':
                            has_car = True
                            break
        except Exception as e:
            print(f"Error reading {filename}: {e}")

        if has_car:
            valid_ids.append(frame_id)
        else:
            skipped_no_car += 1

    print("-" * 30)
    print(f"Total label files scanned: {total_files}")
    print(f"Skipped (Missing Image):   {skipped_no_image}")
    print(f"Skipped (No 'Car'):        {skipped_no_car}")
    print(f"Valid frames to use:       {len(valid_ids)}")
    print("-" * 30)

    if len(valid_ids) == 0:
        print("Error: No valid frames found.")
        return

    # 2. Shuffle
    random.seed(42)
    random.shuffle(valid_ids)

    # 3. Split
    split_index = int(len(valid_ids) * train_ratio)
    train_ids = valid_ids[:split_index]
    val_ids = valid_ids[split_index:]

    print(f"Split Results ({int(train_ratio * 100)} / {int((1 - train_ratio) * 100)})")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")

    # 4. Save
    def save_list(filename, ids):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w') as f:
            f.write("\n".join(ids))
        print(f"Saved: {path}")

    save_list("train.txt", train_ids)
    save_list("val.txt", val_ids)
    save_list("trainval.txt", valid_ids)

    # IMPORTANT: We use the validation set for 'test' to prevent crashes
    save_list("test.txt", val_ids)


if __name__ == "__main__":
    create_splits()