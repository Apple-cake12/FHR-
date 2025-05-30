""" Full assembly of the parts to form the complete network """

from unet_parts import *

class LCResUNet(nn.Module):
    def __init__(self, n_channels, n_classes,kernel_size, bilinear=False):
        super(LCResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel = 64

        self.inc = DoubleConv(n_channels, self.channel,kernel_size)
        self.down1 = Down(self.channel, self.channel * 2,kernel_size)
        self.down2 = Down(self.channel * 2, self.channel * 4,kernel_size)
        self.down3 = Down(self.channel * 4, self.channel * 8,kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channel * 8, self.channel * 16 // factor,kernel_size)
        self.up1 = Up(self.channel * 16, self.channel * 8 // factor,kernel_size, bilinear)
        self.up2 = Up(self.channel * 8, self.channel * 4 // factor,kernel_size, bilinear)
        self.up3 = Up(self.channel * 4, self.channel * 2 // factor,kernel_size, bilinear)
        self.up4 = Up(self.channel * 2, self.channel,kernel_size,bilinear)
        self.outc = OutConv(self.channel, n_classes)
        # Add LC-Residual blocks
        self.res_down_block1 = LCResidualBlock(self.channel)
        self.res_down_block2 = LCResidualBlock(self.channel * 2)
        self.res_down_block3 = LCResidualBlock(self.channel * 4)
        self.res_down_block4 = LCResidualBlock(self.channel * 8)
        self.res_down_block5 = LCResidualBlock(self.channel * 16 // factor)

        self.res_up_block1 = LCResidualBlock(self.channel * 8)
        self.res_up_block2 = LCResidualBlock(self.channel * 4)
        self.res_up_block3 = LCResidualBlock(self.channel * 2)
        self.res_up_block4 = LCResidualBlock(self.channel)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.res_down_block1(x1)
        x2 = self.down1(x1)
        x2 = self.res_down_block2(x2)
        x3 = self.down2(x2)
        x3 = self.res_down_block3(x3)
        x4 = self.down3(x3)
        x4 = self.res_down_block4(x4)
        x5 = self.down4(x4)
        x5 = self.res_down_block5(x5)

        x = self.up1(x5, x4)
        x = self.res_up_block1(x)
        x = self.up2(x, x3)
        x = self.res_up_block2(x)
        x = self.up3(x, x2)
        x = self.res_up_block3(x)
        x = self.up4(x, x1)
        x = self.res_up_block4(x)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

