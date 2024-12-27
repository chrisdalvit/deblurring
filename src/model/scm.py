import torch
import torch.nn as nn

#
# Copied from https://github.com/chosj95/MIMO-UNet/blob/main/models/layers.py
# and https://github.com/chosj95/MIMO-UNet/blob/main/models/MIMOUNet.py
#

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, relu=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.conv(x)
        if self.relu:
            out = self.relu(out)
        return out

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)