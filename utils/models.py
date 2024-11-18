import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn

from torch.utils.data import Dataset
import torch.nn.init as init
import torchvision.models.efficientnet as efficientnet
from torchvision.models.efficientnet import EfficientNet_B1_Weights
from torchvision.models.efficientnet import EfficientNet_B2_Weights
import lpips
import timm

from torchvision import models

    
class SecretEncoder(nn.Module):
    def __init__(self, secret_len, base_res=32, resolution=64):
        super(SecretEncoder, self).__init__()
        
        self.secret_projection = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res), 
            nn.ReLU(inplace=True),
            View(-1, 1, base_res, base_res), 
            ResidualBlock(1, secret_len),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
            nn.Conv2d(secret_len, secret_len, kernel_size=3, padding=1, groups=64),
            nn.ReLU(inplace=True),
            ResidualBlock(secret_len, secret_len//2),
            nn.Conv2d(secret_len//2, 4, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
        )

    def forward(self, x, c):
        secret_map = self.secret_projection(c)     
        x = x + secret_map 
        return secret_map,x



class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        if residual.size(1) != out.size(1):
            residual = self.match_channels(residual)
        return out + residual
    
class SecretDecoder(nn.Module):
    def __init__(self, output_size=64):
        super(SecretDecoder, self).__init__()
        self.output_size = output_size
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size * 2)

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        decoded = self.model(x).view(-1, self.output_size, 2) 
        
        return decoded
    