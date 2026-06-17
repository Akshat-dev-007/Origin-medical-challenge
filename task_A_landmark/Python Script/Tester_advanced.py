import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from Assets.dataset import get_dataloaders
from Assets.model_advanced import AdvancedLandmarkRegressor

# --- Paths ---
IMAGE_DIR = "/content/data/images/images"
CSV_PATH = "/content/data/role_challenge_dataset_ground_truth.csv"
WEIGHTS_PATH = "Model Weights/hypothesis_4_attentive_fpn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

def test_model():
    # 1. Load data and model
    _, val_loader = get_dataloaders(IMAGE_DIR, CSV_PATH, batch_size=BATCH_SIZE)
    val_dataset = val_loader.dataset # To access original image paths
    
    model = AdvancedLandmarkRegressor(num_points=4).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    print("Starting evaluation...")
    
    total_pixel_error_256 = 0.0
    results_list = []
    
    # We use an index counter because val_loader has shuffle=False, 
    # so we can perfectly match predictions back to their original image files.
    sample_idx = 0 
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            
            # 2. Calculate error on the 256x256 training canvas (for our logs)
            preds_px_256 = predictions * 256
            labels_px_256 = labels * 256
            
            preds_pts = preds_px_256.view(-1, 4, 2)
            labels_pts = labels_px_256.view(-1, 4, 2)
            distances = torch.norm(preds_pts - labels_pts, dim=2)
            total_pixel_error_256 += distances.mean().item() * images.size(0)
            
            # 3. Scale predictions back to ORIGINAL dimensions and save
            preds_cpu = predictions.cpu().numpy()
            
            for i in range(images.size(0)):
                # Get the original image path and dimensions
                img_path = val_dataset.image_paths[sample_idx]
                image_name = os.path.basename(img_path)
                orig_w, orig_h = Image.open(img_path).size
                
                # Extract the normalized [0-1] predictions for this specific image
                pred = preds_cpu[i]
                
                # Un-normalize back to original resolution
                ofd_1_x, ofd_2_x = pred[0] * orig_w, pred[2] * orig_w
                ofd_1_y, ofd_2_y = pred[1] * orig_h, pred[3] * orig_h
                bpd_1_x, bpd_2_x = pred[4] * orig_w, pred[6] * orig_w
                bpd_1_y, bpd_2_y = pred[5] * orig_h, pred[7] * orig_h
                
                results_list.append({
                    'image_name': image_name,
                    'ofd_1_x': int(ofd_1_x), 'ofd_1_y': int(ofd_1_y),
                    'ofd_2_x': int(ofd_2_x), 'ofd_2_y': int(ofd_2_y),
                    'bpd_1_x': int(bpd_1_x), 'bpd_1_y': int(bpd_1_y),
                    'bpd_2_x': int(bpd_2_x), 'bpd_2_y': int(bpd_2_y)
                })
                
                sample_idx += 1

    # 4. Final Metrics
    avg_error_256 = total_pixel_error_256 / len(val_dataset)
    
    # Save the dataframe
    df = pd.DataFrame(results_list)
    csv_output_path = "Task_A_Predicted_Landmarks_advanced.csv"
    df.to_csv(csv_output_path, index=False)
    
    print(f"\n--- Test Results ---")
    print(f"Average Radial Error (on 256x256 scale): {avg_error_256:.2f} pixels")
    print(f"Successfully saved all original-scale coordinate predictions to: {csv_output_path}")

if __name__ == "__main__":
    test_model()