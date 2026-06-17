import torch
import torch.nn as nn
import torch.optim as optim
from Assets.dataset import get_dataloaders
from Assets.model import UNet
import os

def train_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    
    # Update these paths when running in Colab
    IMAGE_DIR = "/content/data/images" 
    MASK_DIR = "/content/data/masks"
    SAVE_DIR = "../Model Weights/"
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(IMAGE_DIR, MASK_DIR, batch_size=BATCH_SIZE)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # BCEWithLogitsLoss combines Sigmoid layer and BCELoss in one single class
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Quick validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    # Save the baseline hypothesis model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_1_unet_baseline.pth"))
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_model()