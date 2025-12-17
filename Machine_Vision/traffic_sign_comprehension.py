import os
from logging import getLogger

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import wandb
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint

class SignDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "dataset", batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            ),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.3)
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


    def setup(self, stage=None):
        # Dataset in folder structure: dataset/class_name/*.jpg
        self.dataset = datasets.ImageFolder(self.data_dir)

        # 80/20 split
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_indices, val_indices = torch.utils.data.random_split(
            range(len(self.dataset)), [train_size, val_size]
        )

        self.train_set = torch.utils.data.Subset(
            datasets.ImageFolder(self.data_dir, transform=self.train_transform),
            train_indices
        )

        self.val_set = torch.utils.data.Subset(
            datasets.ImageFolder(self.data_dir, transform=self.val_transform),
            val_indices
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

class TrafficSignCNN(pl.LightningModule):
    def __init__(self, class_names, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.class_names = class_names
        self.num_classes = len(class_names)

        self.model = nn.Sequential(
            # Conv layer Layer 1
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Conv layer Layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # --- Classifier ---
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # --- METRICS ---
        # Initialize metrics for easy calculation
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.confmat = MulticlassConfusionMatrix(num_classes=self.num_classes)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)

        self.log("val_loss", loss, prog_bar=True)

        # Update running metrics
        self.f1.update(preds, y)
        self.confmat.update(preds, y)
        self.accuracy.update(preds, y)

    def on_validation_epoch_end(self):
        # 1. Compute and Log F1
        f1_score = self.f1.compute()
        self.log("val_f1", f1_score, prog_bar=True)

        # 2. Plot and Log Confusion Matrix
        # .plot() returns a Matplotlib Figure and Axis
        fig, ax = self.confmat.plot(labels=self.class_names)

        # Log the figure to WandB as an image
        if self.logger:
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(fig),
                "global_step": self.global_step
            })

        # Close the figure to free memory
        import matplotlib.pyplot as plt
        plt.close(fig)

        # 3. Reset metrics for the next epoch
        self.f1.reset()
        self.confmat.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def predict(self, image):
        traffic_sign_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        pil_img = Image.fromarray(image)
        device = next(self.parameters()).device

        input_tensor = traffic_sign_transform(pil_img).unsqueeze(0)  # shape [1,1,28,28]
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()

        # 4) Map predicted index to class label
        speed = self.class_names[pred_idx]
        if confidence < 0.4:
            speed = None
        return speed

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="sign-cnn")

    data = SignDataModule(data_dir="/home/shared/project/Data/traffic_signs", batch_size=16)
    model = TrafficSignCNN(class_names=["90", "60", "30", "stop"])

    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/shared/project/checkpoints/",
        filename="sign-cnn-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        logger=wandb_logger,
        deterministic=True,
    )

    trainer.fit(model, data)