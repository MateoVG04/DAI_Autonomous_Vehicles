import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class LaneDatasetMultiClass(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.semantic_dir = os.path.join(root_dir, 'semantic')
        self.transform = transform
        self.is_train = is_train
        self.image_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        rgb_path = os.path.join(self.rgb_dir, img_name)
        mask_path = os.path.join(self.semantic_dir, img_name)

        rgb_image = Image.open(rgb_path).convert("RGB")
        semantic_img = Image.open(mask_path)

        # --- 1. Resize BOTH ---
        resize = transforms.Resize((256, 512))
        resize_nearest = transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)

        rgb_image = resize(rgb_image)
        semantic_img = resize_nearest(semantic_img)

        # --- 2. Synchronized Augmentation ---
        if self.is_train:
            if random.random() > 0.5:
                rgb_image = TF.hflip(rgb_image)
                semantic_img = TF.hflip(semantic_img)

        # --- 3. Create Multi-Class Mask ---
        semantic_np = np.array(semantic_img)

        if len(semantic_np.shape) == 3:
            raw_class_mask = semantic_np[:, :, 0]
        else:
            raw_class_mask = semantic_np

        # Initialize Background (Class 0)
        mask = np.zeros_like(raw_class_mask, dtype=np.int64)

        # --- YOUR CORRECTED IDs ---
        # Class 1: Drivable Road
        mask[raw_class_mask == 1] = 1

        # Class 2: Lane Markings
        mask[raw_class_mask == 24] = 2

        # Convert to Tensor
        mask_tensor = torch.from_numpy(mask)

        # --- 4. Transforms ---
        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = transforms.ToTensor()(rgb_image)

        return rgb_image, mask_tensor