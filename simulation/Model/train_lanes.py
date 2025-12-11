import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm  # Progress Bar
from torch.utils.tensorboard import SummaryWriter  # Live Graphs
from torch.cuda.amp import GradScaler, autocast  # Speed up

# Import your custom modules
from model import UNet
from lane_dataset import LaneDataset

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_lane_model(data_root, epochs=50, batch_size=32, learning_rate=1e-4):
    print(f"--- Training on {DEVICE} ---")

    # 1. Setup TensorBoard
    # This creates a folder 'runs/lane_experiment'.
    writer = SummaryWriter('runs/lane_experiment')

    # 2. Transforms & Data
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    # Enable Flipping for Train
    train_ds = LaneDataset(train_dir, transform=train_transform, is_train=True)
    val_ds = LaneDataset(val_dir, transform=val_transform, is_train=False)

    # Pin memory helps transfer data to GPU faster
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # 3. Model & Amp Scaler
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    weights = torch.tensor([1.0, 1.0, 10.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scaler for Mixed Precision (High Speed)
    scaler = GradScaler()

    # 4. Training Loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # --- PROGRESS BAR ---
        # Wraps the loader to show a bar in the terminal
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # --- AUTOMATIC MIXED PRECISION (AMP) ---
            # Runs math in float16 where safe (faster), float32 where needed
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            # Scales loss to prevent underflow in float16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Update progress bar text with current loss
            loop.set_postfix(loss=loss.item())

        # Log average train loss to TensorBoard
        avg_train_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                # Validation doesn't need AMP scaling, but autocast helps speed
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Print simple summary at end of epoch
        print(f" -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- SAVE BEST ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "unet_lanes_road.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print("Training Finished.")


if __name__ == '__main__':
    DATA_PATH = '/mnt/data/carla_dataset'
    train_lane_model(DATA_PATH)