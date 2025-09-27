from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from einops import rearrange, reduce
from jaxtyping import Float, Array

class SimpleCNN(nn.Module):
    __version__ = '0.2.0'

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        # 可选：self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Float[Array, "batch 2 w h"]:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # 如果用于分类，建议不加 sigmoid
        # x = torch.sigmoid(x)
        return x

class SimpleCNNSmall(nn.Module):
    __version__ = '0.1.0'

    def __init__(self):
        super(SimpleCNNSmall, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        # 可选：self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Float[Array, "batch 2 w h"]:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # 如果用于分类，建议不加 sigmoid
        # x = torch.sigmoid(x)
        return x

#TODO: Make kernel larger (to 5 or 7)
class SimpleCNNTiny(nn.Module):
    __version__ = '0.1.0'

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn1 = nn.BatchNorm2d(1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn2 = nn.BatchNorm2d(1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        # 可选：self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Float[Array, "batch 2 w h"]:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

#TODO: Make this smaller
class MultiScale(nn.Module):
    __version__ = '0.1.0'
    
    def __init__(self):
        super(MultiScale, self).__init__()
        
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2, padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1)
        )
        self.conv_3x3_dilated = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1)
        )
        
        self.stem = nn.Sequential(
            nn.Conv2d(int(8*3), 8, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        )
    
    def forward(self, x: Float[Array, "batch 2 w h"]) -> Float[Array, "batch c w h"]:
        features = [
            self.conv_3x3(x),
            self.conv_5x5(x),
            self.conv_3x3_dilated(x),
        ]
        features = rearrange(features, "n b c w h -> b (n c) w h")
        features = self.stem(features)
        return features

# class TinyInception()