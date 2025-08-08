from typing import Tuple
import torch
import torch.nn as nn
from unet_parts import *
from einops import rearrange, reduce
from jaxtyping import Float, Array

class SimpleAutoencoder(nn.Module):
    
    __version__ = '0.1.0'
    
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        self.n_channels = 1
        self.n_classes = 2
        self.bilinear = True
        
        # Encoder
        self.inc = (DoubleConv(self.n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        
        self.tmp_arr = torch.randn((8, 1, 100, 100))
        self.in_dim_ = self.down4(self.down3(self.down2(self.down1(self.inc(self.tmp_arr))))).flatten(1).shape[-1]
        
        # 中间变换层 (保持形状不变)
        # TODO: 变成线性层 
        self.transform = nn.Sequential(
            nn.Linear(self.in_dim_, self.in_dim_ // 8),  # 输入和输出通道数相同
            nn.LeakyReLU(),
            nn.Linear(self.in_dim_ // 8, self.in_dim_ // 8),
            nn.LeakyReLU(),
            nn.Linear(self.in_dim_ // 8, self.in_dim_),
            nn.LeakyReLU(),
        )
        
        # Decoder
        self.up1 = (Up(512, 256 // factor, self.bilinear))
        self.up2 = (Up(256, 128 // factor, self.bilinear))
        self.up3 = (Up(128, 64 // factor, self.bilinear))
        self.up4 = (Up(64, 32, self.bilinear))
        self.outc = (OutConv(32, self.n_classes))

    def forward(self, x: Float[Array, "batch 1 w h"]) -> Tuple[Float[Array, "batch 2 w h"], 
                                                                Float[Array, "batch 1 w h"], 
                                                                Float[Array, "batch h_dim"], 
                                                                Float[Array, "batch h_dim"]]:
        # TODO: 添加残差链接，类似 UNet 结构
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5: torch.Tensor = self.down4(x4)
        
        # y: torch.Tensor = self.transform(x5.flatten(1)).reshape(*x5.shape)
        y: torch.Tensor = self.transform(rearrange(x5, "batch ... -> batch (...)")).reshape(*x5.shape)
        x_tmp = x
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        r_inputs = F.sigmoid(self.outc(x))
        
        x_tmp = self.up1(y, x4)
        x_tmp = self.up2(x_tmp, x3)
        x_tmp = self.up3(x_tmp, x2)
        x_tmp = self.up4(x_tmp, x1)
        logits = F.sigmoid(self.outc(x_tmp))

        return logits,\
            r_inputs, \
            rearrange(x5, "batch ... -> batch (...)"), \
            rearrange(y, "batch ... -> batch (...)")

    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

# class SimpleConvNet(nn.Module):
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
#       
#     def foward(self, x):
#         ...

# 测试网络
if __name__ == "__main__":
    model = SimpleAutoencoder()
    input_tensor = torch.randn(1, 1, 28, 28)  # 示例输入：1个样本，1通道，28x28
    output = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    # 验证输入和输出形状是否相同
    assert input_tensor.shape == output.shape, "输入和输出形状不匹配"
