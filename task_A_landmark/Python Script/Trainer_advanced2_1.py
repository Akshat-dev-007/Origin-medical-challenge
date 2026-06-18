import os
import torch
import torch.optim as optim
from Assets.dataset import get_dataloaders
from Assets.model_advanced import AdvancedLandmarkRegressor
from Assets.losses import LandmarkLoss

# --- Hyperparameters ---
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256  # Our target resize dimension

# --- Loss weights (starting points -- watch the per-component print and rebalance) ---
SIGMA = 2.0        # Gaussian std in heatmap pixels (heatmap is ~64x64 at 256 input)
W_HEATMAP = 10.0
W_COORD = 1.0
W_REG = 0.5

# --- Paths ---
IMAGE_DIR = "/content/data/images/images"
CSV_PATH = "/content/data/role_challenge_dataset_ground_truth.csv"
SAVE_DIR = "Model Weights"

def calculate_pixel_error(preds, targets):
    """Avg Euclidean error in pixels on the 256x256 canvas across all 4 points."""
    preds_px = preds * IMAGE_SIZE
    targets_px = targets * IMAGE_SIZE
    preds_pts = preds_px.view(-1, 4, 2)
    targets_pts = targets_px.view(-1, 4, 2)
    distances = torch.norm(preds_pts - targets_pts, dim=2)
    return distances.mean().item()

def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders(IMAGE_DIR, CSV_PATH, batch_size=BATCH_SIZE)

    model = AdvancedLandmarkRegressor(num_points=4).to(DEVICE)
    criterion = LandmarkLoss(sigma=SIGMA, img_size=IMAGE_SIZE,
                             w_heatmap=W_HEATMAP, w_coord=W_COORD, w_reg=W_REG)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training on {DEVICE}...")
    best_pixel_error = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        agg = {"heatmap": 0.0, "coord": 0.0, "reg": 0.0}

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            predictions, heatmaps = model(images, return_heatmaps=True)
            loss, parts = criterion(predictions, heatmaps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for k in agg:
                agg[k] += parts[k]

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_pixel_error = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                predictions, heatmaps = model(images, return_heatmaps=True)
                loss, _ = criterion(predictions, heatmaps, labels)
                val_loss += loss.item()
                val_pixel_error += calculate_pixel_error(predictions, labels)

        n_tr, n_va = len(train_loader), len(val_loader)
        avg_train_loss = train_loss / n_tr
        avg_val_loss = val_loss / n_va
        avg_pixel_error = val_pixel_error / n_va

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train {avg_train_loss:.4f} "
            f"(hm {agg['heatmap']/n_tr:.4f}, coord {agg['coord']/n_tr:.3f}, reg {agg['reg']/n_tr:.4f}) "
            f"| Val {avg_val_loss:.4f} | Avg Error: {avg_pixel_error:.2f} px"
        )

        # Save the BEST model on validation pixel error, not the last epoch.
        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_4_attentive_fpn.pth"))

    print(f"Training complete. Best Val Avg Error: {best_pixel_error:.2f} px (best checkpoint saved).")

if __name__ == "__main__":
    train_model()