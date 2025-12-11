import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

# Custom Modules
from model import UNet
from lane_dataset_multiclass import LaneDatasetMultiClass

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_multiclass(data_root, epochs=50, batch_size=8, learning_rate=1e-4):
    print(f"--- Training Multi-Class U-Net on {DEVICE} ---")
    writer = SummaryWriter('runs/multiclass_experiment')

    # 1. Transforms (Color only, geometry handled in Dataset)
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

    # 2. Datasets (is_train=True enables flipping)
    train_ds = LaneDatasetMultiClass(train_dir, transform=train_transform, is_train=True)
    val_ds = LaneDatasetMultiClass(val_dir, transform=val_transform, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    print(f"Training on {len(train_ds)} images, Validating on {len(val_ds)}")

    # 3. Model Setup (3 Classes)
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)

    # Weights: Background(0.5), Road(1.0), Lines(10.0) -> Lines are most important
    weights = torch.tensor([0.5, 1.0, 10.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # 4. Training Loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0

        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 5. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        avg_train = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        writer.add_scalar('Loss/Validation', avg_val, epoch)

        # 6. Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "unet_multiclass.pth")
            print(">>> Saved Best Model (unet_multiclass.pth)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training Complete.")


if __name__ == '__main__':
    # POINT THIS TO YOUR DATASET
    DATA_PATH = '/mnt/data/carla_dataset'
    train_multiclass(DATA_PATH)