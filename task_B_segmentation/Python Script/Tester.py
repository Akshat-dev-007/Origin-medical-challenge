import torch
from Assets.dataset import get_dataloaders
from Assets.model import UNet
from Assets.utils import calculate_metrics
import os

def test_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # These paths align with the Colab environment setup
    IMAGE_DIR = "/content/data/images" 
    MASK_DIR = "/content/data/masks"
    WEIGHTS_PATH = "../Model Weights/hypothesis_1_unet_baseline.pth"
    
    # We only need the validation/test loader here
    _, test_loader = get_dataloaders(IMAGE_DIR, MASK_DIR, batch_size=1, val_split=0.2)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Could not find weights at {WEIGHTS_PATH}")
        return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    total_dice = 0.0
    total_iou = 0.0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            
            dice, iou = calculate_metrics(predictions, masks)
            total_dice += dice
            total_iou += iou
            
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    
    print(f"Test Results | Average Dice Score: {avg_dice:.4f} | Average IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    test_model()