import torch

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