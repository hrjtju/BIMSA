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
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1)

    def forward(self, x: Float[Array, "batch 2 w h"]) -> Tuple[Float[Array, "batch 2 w h"], Float[Array, "batch 2 w h"]]:
        x = self.act(self.conv(x)) + x
        x = self.act(self.conv2(x)) + x
        x = self.act(self.conv3(x)) + x
        x = sigmoid(x)
        return x, 1-x