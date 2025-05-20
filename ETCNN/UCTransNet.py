# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm

import torch.nn as nn
import torch
import torch.nn.functional as F
from CTrans import ChannelTransformer


# 确定激活函数
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

# 连续执行nb_Conv次ConvBatchNorm   对应黄线CBRs
def _make_nConv(in_channels, out_channels, nb_Conv,kernel_size, activation='ReLU',):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels,kernel_size=kernel_size,activation=activation))#最少是一次ConvBatchNorm

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels,kernel_size,activation))
    return nn.Sequential(*layers)#使用 *layers 将 layers 列表中的每个元素解包为独立的参数，然后传递给 nn.Sequential 的构造函数。这样，nn.Sequential 就可以正确地将这些层按顺序堆叠起来。


class ConvBatchNorm(nn.Module):#通道不变卷积+归一化+激活
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels,kernel_size, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        # 'same':kernel_size = 2 * padding + 1
        self.conv = nn.Conv1d(in_channels, out_channels,kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = get_activation(activation)#返回激活函数示例
        self._init_weights()

    # 初始化权重
    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):#池化+卷积+Batch+RELUE
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv,kernel_size, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool1d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, kernel_size,activation)

    def forward(self, x):
        out = self.maxpool(x)#池化
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# 通道交叉注意力(CCA)模块
class CCA(nn.Module):
    """
    CCA Block
    """

    def __init__(self, F_g, F_x):#F_g 和 F_x 分别表示输入张量 g 和 x 的通道数。   inchannels//2
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(#nn.Linear(F_x, F_x) 和 nn.Linear(F_g, F_x) 是全连接层，将输入张量的通道数映射到目标通道数 F_x。
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):# g=up x=skip_x    input维度： （batch_size,channels,width）channel可以看成高度
        # channel-wise attention 通道注意力机制
        avg_pool_x = F.avg_pool1d(x, (x.size(2)), stride=(x.size(2)))
        #对输入张量 x 进行全局平均池化，将每个通道的信息压缩为一个标量。（batchsize，channel，1）
        channel_att_x = self.mlp_x(avg_pool_x)#到通道注意力权重。 (batch_size, channels)） 有通道除2的作用

        avg_pool_g = F.avg_pool1d(g, (g.size(2)), stride=(g.size(2)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0#将两个通道注意力权重相加并取平均，得到最终的通道注意力权重

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        #在第 2 维（通道维度）添加一个维度，将权重从 (batch_size, F_x) 转换为 (batch_size, F_x, 1)。
        # 将权重扩展到与输入张量x相同的形状(batch_size, F_x, height, width)。

        x_after_channel = x * scale
        out = self.relu(x_after_channel) #对注意力后的张量应用 ReLU 激活函数，增强非线性特性。
        return out

# 上采样及通道交叉注意力(CCA)模块
class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv,kernel_size, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = get_activation(activation)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv,kernel_size, activation)
        self._init_weights()

    # 初始化权重
    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_out', nonlinearity='relu')

    def forward(self, x, skip_x):
        up = self.up(x)
        up = self.conv(up)
        up = self.norm(up)
        up = self.activation(up) #上采样后的数据  不包括跳跃链接来的
        skip_x_att = self.coatt(g=up, x=skip_x)#CTT来的数据
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension 特征融合

        return self.nConvs(x)


class UCTransNet(nn.Module):
    def __init__(self, config, n_channels, n_classes, kernel_size,img_size=4800, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel

        self.inc = _make_nConv(n_channels, in_channels, nb_Conv=2,kernel_size=kernel_size)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2,kernel_size=kernel_size)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2,kernel_size=kernel_size)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2,kernel_size=kernel_size)
        self.down4 = DownBlock(in_channels * 8, in_channels * 16, nb_Conv=2,kernel_size=kernel_size)
        self.down5 = DownBlock(in_channels * 16, in_channels * 32, nb_Conv=2,kernel_size=kernel_size)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8, in_channels * 16],
                                      patchSize=config.patch_sizes)
        self.up5 = UpBlock_attention(in_channels * 32, in_channels * 16, nb_Conv=2,kernel_size=kernel_size)
        self.up4 = UpBlock_attention(in_channels * 16, in_channels * 8, nb_Conv=2,kernel_size=kernel_size)
        self.up3 = UpBlock_attention(in_channels * 8, in_channels * 4, nb_Conv=2,kernel_size=kernel_size)
        self.up2 = UpBlock_attention(in_channels * 4, in_channels * 2, nb_Conv=2,kernel_size=kernel_size)
        self.up1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2,kernel_size=kernel_size)
        self.outc = nn.Conv1d(in_channels, n_classes, kernel_size=1, stride=1)
        self.last_activation = nn.Softmax(dim=1)
        self._init_weights()

    # 初始化权重
    def _init_weights(self):
        torch.nn.init.kaiming_normal_(self.outc.weight, a=0, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x1, x2, x3, x4, x5, att_weights = self.mtc(x1, x2, x3, x4, x5)

        x = self.up5(x6, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.last_activation(self.outc(x))
        kl_div = F.log_softmax(self.outc(x), dim=-1)#用于计算输入张量的对数 Softmax。这个函数通常用于分类任务中，尤其是在计算交叉熵损失时


        return logits, kl_div

