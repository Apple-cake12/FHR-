""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernel_size, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels,kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,kernel_size)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure x1 and x2 are the same size along the length dimension
        diffX = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LCResidualBlock(nn.Module):
    """LC-Residual block with layer and channel shortcut connections"""
    def __init__(self, in_channels):
        super(LCResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=31, padding=15, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=31, padding=15, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        # Layer shortcut connection
        self.layer_shortcut = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # 添加 Flatten 层
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )
        # Channel shortcut connection
        self.channel_shortcut = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # 添加 Flatten 层
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Layer shortcut
        layer_gate = self.layer_shortcut(out)  #
        out = out * layer_gate.unsqueeze(2)
        # Channel shortcut
        channel_gate = self.channel_shortcut(out)
        out = out * channel_gate.unsqueeze(2)
        out += identity
        out = self.relu(out)
        return out