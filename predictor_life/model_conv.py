import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 输入通道: 1, 输出通道: 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 输入通道: 16, 输出通道: 32
            nn.ReLU()
        )
        
        # 中间变换层 (保持形状不变)
        self.transform = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # 输入和输出通道数相同
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 转置卷积
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 输出通道: 1
            nn.Sigmoid()  # 输出在0-1之间
        )

    def forward(self, x):
        hidden_a = self.encoder(x)
        hidden_b = self.transform(hidden_a)
        pred = self.decoder(hidden_b)
        r_inputs = self.reconstruct(x)
        
        return pred, r_inputs, hidden_a, hidden_b
    
    def reconstruct(self, x):
        return self.decoder(self.encoder(x))

# 测试网络
if __name__ == "__main__":
    model = SimpleAutoencoder()
    input_tensor = torch.randn(1, 1, 28, 28)  # 示例输入：1个样本，1通道，28x28
    output = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    # 验证输入和输出形状是否相同
    assert input_tensor.shape == output.shape, "输入和输出形状不匹配"