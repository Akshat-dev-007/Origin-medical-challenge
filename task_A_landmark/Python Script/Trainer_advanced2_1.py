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
IMAGE_SIZE = 256

# --- Loss config ---
SIGMA = 2.0       # Gaussian std in heatmap pixels (heatmap ~64x64 @ 256 input)
W_COORD = 1.0
W_REG = 0.02      # set to 0.0 first to reproduce the ~27.99 px baseline, then sweep {0.02, 0.05, 0.1}
TEMPERATURE = 1.0 # >1 sharpens the softmax; 1.0 == baseline

# --- Paths ---
IMAGE_DIR = "/content/data/images/images"
CSV_PATH = "/content/data/role_challenge_dataset_ground_truth.csv"
SAVE_DIR = "Model Weights"

def calculate_pixel_error(preds, targets):
    """Avg per-point Euclidean error in pixels on the 256x256 canvas."""
    preds_pts = (preds * IMAGE_SIZE).view(-1, 4, 2)
    targets_pts = (targets * IMAGE_SIZE).view(-1, 4, 2)
    return torch.norm(preds_pts - targets_pts, dim=2).mean().item()

def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_loader, val_loader = get_dataloaders(IMAGE_DIR, CSV_PATH, batch_size=BATCH_SIZE)

    model = AdvancedLandmarkRegressor(num_points=4, temperature=TEMPERATURE).to(DEVICE)
    criterion = LandmarkLoss(sigma=SIGMA, w_coord=W_COORD, w_reg=W_REG)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training on {DEVICE} | w_reg={W_REG} ...")
    best_pixel_error = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        agg = {"coord": 0.0, "reg": 0.0}

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            predictions, prob = model(images, return_heatmaps=True)
            loss, parts = criterion(predictions, prob, labels)
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
                predictions, prob = model(images, return_heatmaps=True)
                loss, _ = criterion(predictions, prob, labels)
                val_loss += loss.item()
                val_pixel_error += calculate_pixel_error(predictions, labels)

        n_tr, n_va = len(train_loader), len(val_loader)
        avg_pixel_error = val_pixel_error / n_va
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train {train_loss/n_tr:.5f} "
            f"(coord {agg['coord']/n_tr:.5f}, reg {agg['reg']/n_tr:.4f}) "
            f"| Val {val_loss/n_va:.5f} | Avg Error: {avg_pixel_error:.2f} px"
        )

        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hypothesis_4_attentive_fpn.pth"))

    print(f"Training complete. Best Val Avg Error: {best_pixel_error:.2f} px (best checkpoint saved).")

if __name__ == "__main__":
    train_model()