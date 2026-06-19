import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from Assets.dataset import get_dataloaders
from Assets.model_advanced import AdvancedLandmarkRegressor  # Using your best model

# --- Paths ---
IMAGE_DIR = "/content/data/images/images"
CSV_PATH = "/content/data/role_challenge_dataset_ground_truth.csv"
WEIGHTS_PATH = "Model Weights/hypothesis_4_attentive_fpn.pth"  # Pointing to the advanced weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

def test_model():
    # 1. Load data and model
    _, val_loader = get_dataloaders(IMAGE_DIR, CSV_PATH, batch_size=BATCH_SIZE)
    val_dataset = val_loader.dataset  # shuffle=False -> index order matches image_paths

    model = AdvancedLandmarkRegressor(num_points=4).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.MSELoss()

    print("Starting evaluation...")

    total_pixel_error_256 = 0.0   # avg per-point radial error on the 256x256 canvas
    total_pixel_error_orig = 0.0  # avg per-point radial error on the native resolution
    total_mse_loss = 0.0
    results_list = []

    sample_idx = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)  # normalized [0,1], default forward path

            # 2a. Normalized MSE
            total_mse_loss += criterion(predictions, labels).item() * images.size(0)

            # 2b. Radial error on the 256x256 canvas
            preds_pts_256 = (predictions * 256).view(-1, 4, 2)
            labels_pts_256 = (labels * 256).view(-1, 4, 2)
            distances_256 = torch.norm(preds_pts_256 - labels_pts_256, dim=2)  # [B, 4]
            total_pixel_error_256 += distances_256.mean().item() * images.size(0)

            # 3. Per-sample: un-normalize to ORIGINAL dimensions, score, and save
            preds_cpu = predictions.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            for i in range(images.size(0)):
                img_path = val_dataset.image_paths[sample_idx]
                image_name = os.path.basename(img_path)
                orig_w, orig_h = Image.open(img_path).size

                pred = preds_cpu[i]
                gt = labels_cpu[i]

                # Radial error in ORIGINAL pixels (the clinically meaningful number).
                # x scales by orig_w, y by orig_h -- the same anisotropic scaling used in the dataset.
                dx = (pred[0::2] - gt[0::2]) * orig_w
                dy = (pred[1::2] - gt[1::2]) * orig_h
                total_pixel_error_orig += np.mean(np.sqrt(dx ** 2 + dy ** 2))

                # Un-normalize predictions back to original resolution (round, don't truncate)
                ofd_1_x, ofd_2_x = pred[0] * orig_w, pred[2] * orig_w
                ofd_1_y, ofd_2_y = pred[1] * orig_h, pred[3] * orig_h
                bpd_1_x, bpd_2_x = pred[4] * orig_w, pred[6] * orig_w
                bpd_1_y, bpd_2_y = pred[5] * orig_h, pred[7] * orig_h

                results_list.append({
                    'image_name': image_name,
                    'ofd_1_x': int(round(ofd_1_x)), 'ofd_1_y': int(round(ofd_1_y)),
                    'ofd_2_x': int(round(ofd_2_x)), 'ofd_2_y': int(round(ofd_2_y)),
                    'bpd_1_x': int(round(bpd_1_x)), 'bpd_1_y': int(round(bpd_1_y)),
                    'bpd_2_x': int(round(bpd_2_x)), 'bpd_2_y': int(round(bpd_2_y))
                })

                sample_idx += 1

    # 4. Final Metrics
    n = len(val_dataset)
    avg_error_256 = total_pixel_error_256 / n
    avg_error_orig = total_pixel_error_orig / n
    avg_mse = total_mse_loss / n

    df = pd.DataFrame(results_list)
    csv_output_path = "Task_A_Predicted_Landmarks_advanced.csv"
    df.to_csv(csv_output_path, index=False)

    print(f"\n--- Test Results ---")
    print(f"Average Normalized MSE:                     {avg_mse:.6f}")
    print(f"Average Radial Error (256x256 canvas):      {avg_error_256:.2f} px")
    print(f"Average Radial Error (original resolution): {avg_error_orig:.2f} px")
    print(f"Saved original-scale predictions to:        {csv_output_path}")

if __name__ == "__main__":
    test_model()