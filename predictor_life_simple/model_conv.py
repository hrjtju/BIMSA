from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from einops import rearrange, reduce
from jaxtyping import Float, Array

class SimpleCNN(nn.Module):
    
    __version__ = '0.1.0'
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 2, kernel_size=1, stride=1)

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]:
        x = self.conv(x)
        x = self.act(x)
        x = self.conv2(x)
        x = sigmoid(x)
        return x, 1-x

class SimpleCNN3Res(nn.Module):
    
    __version__ = '0.1.0'
    
    def __init__(self):
        super(SimpleCNN3Res, self).__init__()

        self.conv = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.conv3 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, padding_mode="circular")

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Tuple[Float[Array, "batch 1 w h"], Float[Array, "batch 1 w h"]]:
        y = self.act(self.conv(x))
        y = self.act(self.conv2(y))
        y = self.act(self.conv3(y)) + x
        y = sigmoid(y)
        y = y
        return y, 1-y

class MultiScale(nn.Module):
    def __init__(self):
        super(MultiScale, self).__init__()
        
        self.feature_extractor_ls = {
            "3x3": nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            "5x5": nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=1, padding_mode="circular"),
            "7x7": nn.Conv2d(2, 128, kernel_size=7, stride=1, padding=1, padding_mode="circular"),
            "3x3_dilated": nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode="circular"),
            "5x5_dilated": nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2, dilation=2, padding_mode="circular"),
            "7x7_dilated": nn.Conv2d(2, 128, kernel_size=7, stride=1, padding=2, dilation=2, padding_mode="circular"),
            "3x3_dilated_2": nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=4, dilation=4, padding_mode="circular"),
            "5x5_dilated_2": nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=4, dilation=4, padding_mode="circular"),
            "7x7_dilated_2": nn.Conv2d(2, 128, kernel_size=7, stride=1, padding=4, dilation=4, padding_mode="circular"),
        }
        self.act = nn.SiLU()
    
    def forward(self, x: Float[Array, "batch 2 w h"]) -> Float[Array, "batch c w h"]:
        features = []
        for key in self.feature_extractor_ls:
            features.append(self.act(self.feature_extractor_ls[key](x)))
        features = torch.cat(features, dim=1)
        return features
