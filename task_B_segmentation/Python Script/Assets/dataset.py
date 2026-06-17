import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import random
from sklearn.model_selection import train_test_split

class FetalUltrasoundDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Load image and mask
        image = Image.open(self.image_paths[idx]).convert("L") 
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # --- NEW CONTOUR FILLING LOGIC since the annotation is an edge map not solid mask---
        # Convert PIL mask to numpy array for OpenCV processing
        mask_np = np.array(mask)
        
        # Find all contours in the thin-line annotation
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank black mask of the same size
        filled_mask_np = np.zeros_like(mask_np)
        
        # Draw the contours onto the blank mask. thickness=-1 tells OpenCV to fill the shape.
        cv2.drawContours(filled_mask_np, contours, -1, (255), thickness=-1)
        
        # Convert the filled numpy array back to a PIL Image
        mask = Image.fromarray(filled_mask_np)
        # ---------------------------------
        # 1. Resize (Required for both train and val)
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        # 2. Augmentations (Only for training set)
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 3. Convert to Tensor and normalize
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Binarize mask just in case of interpolation artifacts
        mask = (mask > 0.5).float()

        return image, mask

def get_dataloaders(image_dir, mask_dir, batch_size=8, val_split=0.2):
    # Sort to ensure images and masks align
    all_images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    all_masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    # Split raw data BEFORE augmentation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=val_split, random_state=42
    )
    
    train_dataset = FetalUltrasoundDataset(train_imgs, train_masks, is_train=True)
    val_dataset = FetalUltrasoundDataset(val_imgs, val_masks, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader