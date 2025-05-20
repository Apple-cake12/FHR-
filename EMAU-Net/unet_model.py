""" Full assembly of the parts to form the complete network """

from unet_parts import *

class MAU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size, bilinear=False):
        super(MAU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.channel = 64

        # Encoder
        self.inc = DoubleConv(n_channels, self.channel,kernels_size=kernel_size, mid_channels=self.channel)
        self.down1 = Down(self.channel, self.channel * 2, kernels_size=kernel_size,use_eca=True)
        self.down2 = Down(self.channel * 2, self.channel * 4,kernels_size=kernel_size, use_eca=True)
        self.down3 = Down(self.channel * 4, self.channel * 8,kernels_size=kernel_size, use_eca=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.channel * 8, self.channel * 16 // factor,kernels_size=kernel_size, use_eca=True, use_nla=True)

        # Decoder
        self.up1 = Up(self.channel * 16, self.channel * 8 // factor,kernels_size=kernel_size, bilinear=bilinear)
        self.up2 = Up(self.channel * 8, self.channel * 4 // factor, kernels_size=kernel_size, bilinear=bilinear)
        self.up3 = Up(self.channel * 4, self.channel * 2 // factor, kernels_size=kernel_size, bilinear=bilinear)
        self.up4 = Up(self.channel * 2, self.channel, kernels_size=kernel_size, bilinear=bilinear)
        self.outc = OutConv(self.channel, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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

