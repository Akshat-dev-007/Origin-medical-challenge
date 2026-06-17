import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, labels_df, target_size=(256, 256)):
        self.image_paths = image_paths
        self.labels_df = labels_df.set_index('image_name')
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        
        # 1. Load image
        image = Image.open(img_path).convert("L")
        orig_w, orig_h = image.size
        
        # 2. Get the original coordinates
        coords = self.labels_df.loc[image_name].values.astype(np.float32)
        
        # 3. Normalize coordinates to [0.0, 1.0]
        # Divide X by orig_w, Y by orig_h
        coords[0::2] = coords[0::2] / orig_w
        coords[1::2] = coords[1::2] / orig_h
        
        # 4. Resize image and convert to tensor
        image = TF.resize(image, self.target_size)
        image = TF.to_tensor(image)
        
        # Convert to torch tensor
        labels = torch.tensor(coords, dtype=torch.float32)

        return image, labels

def get_dataloaders(image_dir, csv_path, batch_size=16, val_split=0.2):
    df = pd.read_csv(csv_path)
    
    valid_image_names = set(df['image_name'].tolist())
    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f in valid_image_names and f.endswith(('.png', '.jpg'))]
    
    all_images.sort()
    print(f"Found {len(all_images)} images matching the CSV records.")

    train_imgs, val_imgs = train_test_split(all_images, test_size=val_split, random_state=42)
    
    train_dataset = LandmarkDataset(train_imgs, df)
    val_dataset = LandmarkDataset(val_imgs, df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader