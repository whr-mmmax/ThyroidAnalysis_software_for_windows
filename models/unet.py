import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础模块

class DoubleConv(nn.Module):
    """两个卷积块，每个包含 Conv2d -> BN -> ReLU。"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """下采样：最大池化 + DoubleConv。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样，拼接跳跃连接，再经过 DoubleConv。"""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐空间尺寸
        dY = x2.size(2) - x1.size(2)
        dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionGate(nn.Module):
    """软注意力门（Oktay et al.）"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# U-Net 主网络

class UNet(nn.Module):
    """
    U-Net 分割网络，支持注意力门。
    参数:
        n_channels   : 输入通道数（灰度图=1）
        n_classes    : 输出类别数（二值分割=1）
        bilinear     : 是否使用双线性上采样
        use_attention: 是否在跳跃连接处加入注意力门
    """
    def __init__(self,
                 n_channels:   int  = 1,
                 n_classes:    int  = 1,
                 bilinear:     bool = True,
                 use_attention:bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_attention = use_attention

        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # 注意力门（如果启用）
        if use_attention:
            self.att1 = AttentionGate(F_g=1024 // factor, F_l=512, F_int=256)
            self.att2 = AttentionGate(F_g=512 // factor, F_l=256, F_int=128)
            self.att3 = AttentionGate(F_g=256 // factor, F_l=128, F_int=64)
            self.att4 = AttentionGate(F_g=128 // factor, F_l=64, F_int=32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder 前向
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder 前向（可选注意力）
        skip4 = self.att1(x5, x4) if self.use_attention else x4
        x = self.up1(x5, skip4)

        skip3 = self.att2(x, x3) if self.use_attention else x3
        x = self.up2(x, skip3)

        skip2 = self.att3(x, x2) if self.use_attention else x2
        x = self.up3(x, skip2)

        skip1 = self.att4(x, x1) if self.use_attention else x1
        x = self.up4(x, skip1)

        return self.outc(x)

    def predict(self, x, threshold: float = 0.5) -> torch.Tensor:
        """返回二值化掩码（0/1）。"""
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits)
            return (prob > threshold).float()