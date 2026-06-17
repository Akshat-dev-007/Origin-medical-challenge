import torch
import torch.nn as nn
import torchvision.models as models

class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks=8):
        super(LandmarkRegressor, self).__init__()
        # Load pre-trained ResNet34
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify the first layer for 1-channel grayscale
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).conv1.weight.data.sum(dim=1, keepdim=True)

        
        # Replace the classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_landmarks),
            nn.Sigmoid() # Forces outputs strictly between 0.0 and 1.0
        )

    def forward(self, x):
        return self.backbone(x)