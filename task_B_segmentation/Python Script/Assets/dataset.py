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
    # 1. Get all files and filter out hidden files like .DS_Store
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 2. Map files to their base ID (e.g., '000_HC.png' -> '000_HC' and '000_HC_Annotation.png' -> '000_HC')
    # This handles the '_Annotation' suffix dynamically
    image_dict = {f.split('.')[0].replace('_HC', ''): f for f in image_files}
    mask_dict = {f.split('.')[0].replace('_HC', '').replace('_Annotation', ''): f for f in mask_files}

    # 3. Find the intersection of IDs present in both folders
    common_ids = sorted(list(set(image_dict.keys()).intersection(set(mask_dict.keys()))))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks.")
    print(f"Successfully matched {len(common_ids)} pairs based on Subject IDs.")

    # 4. Rebuild matching, ordered path lists
    image_paths = [os.path.join(image_dir, image_dict[cid]) for cid in common_ids]
    mask_paths = [os.path.join(mask_dir, mask_dict[cid]) for cid in common_ids]

    # 5. Perform the train/val split safely
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )

    # 6. Instantiate datasets
    train_dataset = FetalUltrasoundDataset(train_imgs, train_masks, is_train=True)
    val_dataset = FetalUltrasoundDataset(val_imgs, val_masks, is_train=False)

    # 7. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader