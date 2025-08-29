import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------
#         基础组件
# -----------------------------------------------



# -----------------------------------------------
#         ResNet 模块和诸部件
# -----------------------------------------------


# -----------------------------------------------
#         UNet 模块和诸部件
# -----------------------------------------------
class UNet_DoubleConv(nn.Module):
    """
    UNet 的双卷积组件。
    
    结构：(convolution => [BN] => ReLU) * 2
    
    参数：
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    mid_channels (int, optional): 中间通道数，默认与输出通道数相同
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 mid_channels: int | None = None
                 ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet_DownSampleBlock(nn.Module):
    """
    UNet 的下采样模块
    结构：MaxPool2d => DoubleConv
    
    参数：
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            UNet_DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UNet_UpSampleBlock(nn.Module):
    """
    UNet 的上采样模块
    结构：Upsample/ConvTranspose2d => UNet_DoubleConv
    
    参数：
    in_channels (int): 输入通道数
    out_channels (int): 输出通道数
    bilinear (bool, optional): 是否使用双线性插值上采样，默认值为 True
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNet_DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNet_DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -----------------------------------------------
#         多尺度特征融合模块
# -----------------------------------------------

class MFF(nn.Module):
    """
    多尺度特征融合模块。
    
    该模块接收多个不同尺度的特征图，调整它们的通道数和空间尺寸，然后将它们融合为一个统一的特征图。
    
    参数：
    in_channels_list (list of int): 输入特征图的通道数列表
    out_channels (int): 输出特征图的通道数
    target_size (tuple of int): 目标空间尺寸 (height, width)
    """
    def __init__(self, in_channels_list, out_channels, target_size):
        super(MFF, self).__init__()
        self.target_size = target_size
        self.adjust_layers = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.adjust_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.Upsample(size=target_size, mode='bilinear', align_corners=True)
                )
            )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_maps):
        adjusted_maps = []
        for i, feature_map in enumerate(feature_maps):
            adjusted_map = self.adjust_layers[i](feature_map)
            adjusted_maps.append(adjusted_map)
        
        fused_map = torch.cat(adjusted_maps, dim=1)
        fused_map = self.fusion_conv(fused_map)
        
        return fused_map

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模块。
    
    该模块通过自适应地重新校准通道间的特征响应，提升模型的表示能力。
    
    参数：
    channels (int): 输入特征图的通道数
    reduction (int, optional): 通道数缩减比例，默认值为 16
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Feature_PyramidBlock(nn.Module):
    """
    特征金字塔模块。
    
    该模块通过不同尺度的卷积和池化操作，提取不同层级的特征图。
    
    参数：
    in_channels (int): 输入特征图的通道数
    out_channels (int): 输出特征图的通道数
    """
    def __init__(self, in_channels, out_channels):
        super(Feature_PyramidBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        p1 = self.pool(x2)
        p1 = self.se_block(p1)
        
        p2 = self.pool(p1)
        p2 = self.se_block(p2)
        
        return x2, p1, p2

# -----------------------------------------------
#         损失函数
# -----------------------------------------------

class IoU_Loss(nn.Module):
    """
    用于二分类任务的 IoU 损失函数。
    计算预测与目标之间的交并比，并返回 1 - IoU 作为损失值。
    
    参数：
    smooth (float): 平滑项，防止除零错误。默认值为 1e-6。
    """
    def __init__(self, smooth: float = 1e-6):
        super(IoU_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        total = (preds + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class Dice_Loss(nn.Module):
    """
    用于二分类任务的 Dice 损失函数。
    计算预测与目标之间的 Dice 系数，并返回 1 - Dice 作为损失值。
    
    参数：
    smooth (float): 平滑项，防止除零错误。默认值为 1e-6。
    """
    def __init__(self, smooth: float = 1e-6):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class Focal_Loss(nn.Module):
    """
    用于二分类任务的 Focal 损失函数。
    通过调整易分类样本和难分类样本的权重，来处理类别不平衡问题。
    
    参数：
    alpha (float): 平衡因子，用于调整正负样本的权重。默认值为 0.25。
    gamma (float): Focusing 参数，用于调整易分类样本的权重。默认值为 2.0。
    reduction (str): 损失值的计算方式，可选 'mean'、'sum' 或 'none'。默认值为 'mean'。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# -----------------------------------------------
#         其他工具函数
# -----------------------------------------------

