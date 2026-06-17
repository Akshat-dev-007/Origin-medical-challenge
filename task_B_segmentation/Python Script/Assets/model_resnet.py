import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Upsample the input map by a factor of 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Convolutions after concatenating with the skip connection
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Concatenate along the channel dimension
            x = torch.cat([skip, x], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResNet34UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNet34UNet, self).__init__()
        # Load the pre-trained ResNet34 backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Adapt the first convolution to accept 1-channel (grayscale) instead of 3-channel (RGB)
        # We reuse the pre-trained weights by summing them across the color channels
        self.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstconv.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        # Encoder layers
        self.encoder1 = resnet.layer1 # Outputs 64 channels
        self.encoder2 = resnet.layer2 # Outputs 128 channels
        self.encoder3 = resnet.layer3 # Outputs 256 channels
        self.encoder4 = resnet.layer4 # Outputs 512 channels
        
        # Decoder layers (in_channels, skip_channels, out_channels)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        
        # Final upsampling to restore the original 256x256 image size
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder path with skip connections
        x0 = self.firstrelu(self.firstbn(self.firstconv(x))) # Skip 0
        x1 = self.encoder1(self.firstmaxpool(x0))            # Skip 1
        x2 = self.encoder2(x1)                               # Skip 2
        x3 = self.encoder3(x2)                               # Skip 3
        x4 = self.encoder4(x3)                               # Bottleneck
        
        # Decoder path
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        
        out = self.final_conv(d1)
        return out