import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def check_mask_filling(image_path, mask_path):
    # 1. Load raw images
    print(f"Loading Image: {image_path}")
    raw_image = Image.open(image_path).convert("L")
    raw_mask = Image.open(mask_path).convert("L")

    # 2. Apply our contour filling logic from dataset.py
    mask_np = np.array(raw_mask)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_mask_np = np.zeros_like(mask_np)
    cv2.drawContours(filled_mask_np, contours, -1, (255), thickness=-1)

    # 3. Create a colored overlay for visual confirmation
    # Convert grayscale ultrasound to an RGB array
    overlay = np.stack((np.array(raw_image),)*3, axis=-1)
    
    # Where the filled mask is white, blend a red tint (50% opacity)
    red_mask = np.zeros_like(overlay)
    red_mask[filled_mask_np == 255] = [255, 0, 0]
    
    alpha = 0.4
    mask_indices = filled_mask_np == 255
    overlay[mask_indices] = (1 - alpha) * overlay[mask_indices] + alpha * red_mask[mask_indices]

    # 4. Plot the pipeline
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].set_title('1. Original Ultrasound')
    axes[0].axis('off')

    axes[1].imshow(raw_mask, cmap='gray')
    axes[1].set_title('2. Original Annotation (Contour)')
    axes[1].axis('off')

    axes[2].imshow(filled_mask_np, cmap='gray')
    axes[2].set_title('3. Processed Mask (Filled)')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('4. Final Overlay Check')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure these point to where you saved the two files you just uploaded
    sample_img = "../../000_HC.png"  
    sample_mask = "../../000_HC_Annotation.png"
    
    if os.path.exists(sample_img) and os.path.exists(sample_mask):
        check_mask_filling(sample_img, sample_mask)
    else:
        print(f"Error: Could not find files at {sample_img} or {sample_mask}.")
        print("Please place the sample images in the correct directory or update the paths.")