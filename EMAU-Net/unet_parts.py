""" Parts of the U-Net model """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,kernels_size, mid_channels=None, use_eca=False, use_nla=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernels_size, padding=kernels_size//2, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernels_size, padding=kernels_size//2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 添加注意力模块
        self.use_eca = use_eca
        self.use_nla = use_nla
        if use_eca:
            self.eca = ECA(out_channels)
        if use_nla:
            self.nla = NLA(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_eca:
            x = self.eca(x)
        if self.use_nla:
            x = self.nla(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,kernels_size, use_eca=False, use_nla=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels,kernels_size=kernels_size, use_eca=use_eca, use_nla=use_nla)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,kernels_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,kernels_size=kernels_size)

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

class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=-1, keepdim=True)  # Global Average Pooling
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class NLA(nn.Module):
    """Non-Local Attention module"""
    def __init__(self, in_channels):
        super(NLA, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width = x.size()
        proj_query = self.query(x).view(m_batchsize, -1, width).permute(0, 2, 1)
        proj_key = self.key(x).view(m_batchsize, -1, width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(m_batchsize, -1, width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width)
        out = self.gamma * out + x
        return out