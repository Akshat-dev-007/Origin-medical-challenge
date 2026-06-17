import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Assets.dataset import get_dataloaders
from Assets.model import LandmarkRegressor

# --- Hyperparameters ---
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256 # Our target resize dimension

# --- Paths ---
IMAGE_DIR = "/content/data/images/images" 
CSV_PATH = "/content/data/role_challenge_dataset_ground_truth.csv" # Update if your CSV has a different name or path!
SAVE_DIR = "Model Weights"

def calculate_pixel_error(preds, targets):
    """
    Converts 0-1 normalized predictions back to 256x256 pixel coordinates
    and calculates the average Euclidean distance error across all 4 points.
    """
    # Un-normalize to 256x256 space
    preds_px = preds * IMAGE_SIZE
    targets_px = targets * IMAGE_SIZE
    
    # Reshape from [Batch, 8] to [Batch, 4 points, 2 coords (x,y)]
    preds_pts = preds_px.view(-1, 4, 2)
    targets_pts = targets_px.view(-1, 4, 2)
    
    # Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
    distances = torch.norm(preds_pts - targets_pts, dim=2)
    
    # Return average distance in pixels
    return distances.mean().item()

def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders(IMAGE_DIR, CSV_PATH, batch_size=BATCH_SIZE)
    
    model = LandmarkRegressor(num_landmarks=8).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        val_pixel_error = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                predictions = model(images)
                
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                val_pixel_error += calculate_pixel_error(predictions, labels)
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_pixel_error = val_pixel_error / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | Avg Error: {avg_pixel_error:.2f} px")

    # Save the final weights
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_3_resnet_regression.pth"))
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_model()