import torch
import cv2
import numpy as np

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    # Apply sigmoid to convert raw logits to probabilities
    pred_probs = torch.sigmoid(pred_mask)
    
    # Binarize the predictions based on the threshold
    pred_binary = (pred_probs > threshold).float()
    
    # Flatten the tensors to 1D arrays for easy computation
    pred_flat = pred_binary.view(-1)
    true_flat = true_mask.view(-1)
    
    intersection = (pred_flat * true_flat).sum()
    union = pred_flat.sum() + true_flat.sum() - intersection
    
    # Add a small epsilon (1e-6) to prevent division by zero
    # Dice = 2 * Intersection / (Area 1 + Area 2)
    dice = (2. * intersection + 1e-6) / (pred_flat.sum() + true_flat.sum() + 1e-6)
    
    # IoU = Intersection / Union
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice.item(), iou.item()


def extract_biometry_points(binary_mask_np):
    """
    Takes a 2D numpy array binary mask (0s and 1s) and returns the 4 biometry points.
    Returns: dict with 'a', 'b', 'c', 'd' coordinate tuples, or None if no contour found.
    """
    # Ensure mask is uint8 for OpenCV
    mask_uint8 = (binary_mask_np * 255).astype(np.uint8)
    
    # 1. Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # 2. Get the largest contour (the cranium)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Need at least 5 points to fit an ellipse
    if len(largest_contour) < 5:
        return None
        
    # 3. Fit an optimal ellipse
    # Returns (center(x, y), (minor_axis_length, major_axis_length), angle)
    ellipse = cv2.fitEllipse(largest_contour)
    (cx, cy), (width, height), angle = ellipse
    
    # Convert angle to radians for trig math
    # OpenCV's fitEllipse angle is the rotation of the major axis from the vertical
    rad = np.deg2rad(angle)
    
    # 4. Calculate Major Axis endpoints (b, d) - Occipitofrontal Diameter (OFD)
    dx_maj = (height / 2) * np.sin(rad)
    dy_maj = -(height / 2) * np.cos(rad)
    
    b = (int(cx + dx_maj), int(cy + dy_maj))
    d = (int(cx - dx_maj), int(cy - dy_maj))
    
    # 5. Calculate Minor Axis endpoints (a, c) - Biparietal Diameter (BPD)
    dx_min = (width / 2) * np.cos(rad)
    dy_min = (width / 2) * np.sin(rad)
    
    a = (int(cx + dx_min), int(cy + dy_min))
    c = (int(cx - dx_min), int(cy - dy_min))
    
    return {
        'ellipse_center': (int(cx), int(cy)),
        'a': a, 'c': c, # Minor axis (BPD)
        'b': b, 'd': d  # Major axis (OFD)
    }