# Origin Medical - Fetal Ultrasound Biometry Extraction
## Executive Summary
This repository contains a complete, end-to-end deep learning and computer vision pipeline for extracting fetal biometry points (Biparietal Diameter and Occipitofrontal Diameter) from ultrasound imagery. The challenge was tackled via two distinct paradigms to evaluate the optimal trade-off between geometric inference and direct regression:

## Task B (Segmentation-to-Geometry): Utilizing semantic segmentation to isolate the cranium, followed by classical computer vision algorithms to mathematically extract the axes.

## Task A (Direct Regression): Training deep neural networks to directly predict continuous spatial coordinates using normalized bounding and advanced feature pyramid fusion.

Repository Structure
The codebase is fully modular, allowing independent training and evaluation of different architectures without overlapping dependencies.

Plaintext
Origin-medical-challenge/
├── task_1_segmentation/         # Task B Pipeline
│   ├── Python Script/
│   │   ├── Assets/
│   │   │   ├── dataset.py       # ID-matching data loader
│   │   │   ├── model.py         # Baseline Vanilla U-Net (Hypothesis 1)
│   │   │   ├── model_resnet.py  # ResNet-34 U-Net (Hypothesis 2)
│   │   │   └── utils.py         # OpenCV biometry extraction logic
│   │   ├── Trainer.py
│   │   ├── Tester.py            # Generates Dice/IoU & CSV outputs
│   ├── Model Weights/           # Contains .pth weight files
│   └── Task_B_Predicted_Landmarks_resnet34.csv
│
├── task_1_landmark/             # Task A Pipeline
│   ├── Python Script/
│   │   ├── Assets/
│   │   │   ├── dataset.py       # Coordinate scaling & normalization loader
│   │   │   ├── model.py         # ResNet-34 Linear Regressor (Hypothesis 3)
│   │   │   └── model_advanced.py# Attentive FPN & Soft-Argmax (Hypothesis 4)
│   │   ├── Trainer.py
│   │   ├── Tester.py            # Generates MSE/Radial Error & CSV outputs
│   ├── Model Weights/           # Contains .pth weight files
│   └── Task_A_Predicted_Landmarks_advanced.csv
Data Engineering & Leakage Prevention
To ensure robust, production-ready evaluation, strict data isolation protocols were enforced across both tasks.

Guaranteed Alignment: For segmentation, images and masks were dynamically paired by parsing their base Subject IDs (e.g., 000_HC).

Leak-Free Validation: The train/validation split was executed strictly on the raw list of subject identifiers before any tensor conversions, augmentations, or scaling factors were applied. This explicitly prevents cross-contamination and ensures the validation metrics reflect true generalization.

Pandas Optimization: All output aggregation during evaluation relies on building Python lists and executing a single pd.DataFrame() creation at the end, explicitly avoiding the deprecated and computationally expensive .append() method on DataFrames.

Task B: Segmentation & Computer Vision Extraction
Methodology: The objective was to generate a highly accurate binary mask of the cranium and mathematically derive the 4 biometry points.

Hypothesis 1 (Vanilla U-Net): A standard encoder-decoder architecture was trained from scratch.

Result: 0.9585 Dice Score | 0.9287 IoU

Hypothesis 2 (ResNet-34 U-Net): To handle acoustic shadowing and ultrasound speckle noise, the encoder was replaced with pre-trained ImageNet weights from a ResNet-34 backbone. The pre-learned edge detection resulted in faster convergence and sharper mask boundaries.

Result: 0.9635 Dice Score | 0.9370 IoU

Geometric Extraction (utils.py): The resulting predicted masks were converted to uint8 arrays. OpenCV was utilized to find the largest external contour and fit an optimal ellipse. Using the angle of rotation returned by the ellipse fitting, the trigonometric endpoints of the major axis (OFD) and minor axis (BPD) were dynamically calculated and mapped back to absolute pixel coordinates.

Task A: Direct Coordinate Regression
Methodology: The objective was to predict the 8 continuous spatial coordinates directly from the raw image. To stabilize training gradients and prevent out-of-bounds predictions, labels were converted from absolute pixels to relative normalized values [0.0, 1.0] by dividing by the original image dimensions before resizing.

Hypothesis 3 (ResNet-34 Linear Regression): A pre-trained ResNet-34 backbone capped with a Fully Connected linear head and a Sigmoid activation.

Result: 29.00 px Average Radial Error (on a 256x256 scaled canvas).

Hypothesis 4 (Attentive FPN with Regression Voting): Flattening a 2D image into dense linear nodes destroys spatial geometry. To resolve this, a custom architecture was engineered. It utilizes a Feature Pyramid Network (FPN) to fuse multi-scale features, applies Spatial Attention, and outputs 4 heatmaps. A SoftArgmax2D module was implemented so pixels "vote" probabilistically on the coordinate locations, preserving the 2D matrix throughout the entire network.

Result: 27.99 px Average Radial Error.

Execution Instructions
To replicate the final CSV outputs using the saved model weights:

For Task B (Segmentation):

Bash
python task_1_segmentation/Python Script/Tester.py
Outputs: Task_B_Predicted_Landmarks_resnet34.csv

For Task A (Regression):

Bash
python task_1_landmark/Python Script/Tester.py