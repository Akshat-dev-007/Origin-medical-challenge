import torch
import numpy as np
from Assets.dataset import get_dataloaders
from Assets.model import UNet
from Assets.utils import calculate_metrics, extract_biometry_points
import os

def test_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # These paths align with the Colab environment setup
    IMAGE_DIR = "/content/data/images/images" 
    MASK_DIR = "/content/data/masks/masks"
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
    # Create a list to hold our prediction dictionaries
    results_list = []
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            
            dice, iou = calculate_metrics(predictions, masks)
            total_dice += dice
            total_iou += iou
            
            # Convert prediction to binary numpy array
            pred_probs = torch.sigmoid(predictions[0]).cpu().numpy().squeeze()
            binary_mask = (pred_probs > 0.5).astype(np.uint8)
            
            landmarks = extract_biometry_points(binary_mask)
            
            # Store the results. If the model fails to find a contour, we log None.
            if landmarks:
                results_list.append({
                    'Image_Index': idx,
                    'BPD_a_x': landmarks['a'][0], 'BPD_a_y': landmarks['a'][1],
                    'BPD_c_x': landmarks['c'][0], 'BPD_c_y': landmarks['c'][1],
                    'OFD_b_x': landmarks['b'][0], 'OFD_b_y': landmarks['b'][1],
                    'OFD_d_x': landmarks['d'][0], 'OFD_d_y': landmarks['d'][1],
                    'Dice_Score': round(dice, 4)
                })
            else:
                results_list.append({
                    'Image_Index': idx,
                    'BPD_a_x': None, 'BPD_a_y': None, 'BPD_c_x': None, 'BPD_c_y': None,
                    'OFD_b_x': None, 'OFD_b_y': None, 'OFD_d_x': None, 'OFD_d_y': None,
                    'Dice_Score': round(dice, 4)
                })
                
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    
    # Save the dataframe to a CSV file
    import pandas as pd
    df = pd.DataFrame(results_list)
    csv_path = "Task_B_Predicted_Landmarks.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Test Results | Average Dice Score: {avg_dice:.4f} | Average IoU: {avg_iou:.4f}")
    print(f"Successfully saved all coordinate predictions to: {csv_path}")

if __name__ == "__main__":
    test_model()