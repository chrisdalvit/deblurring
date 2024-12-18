import torch
import torch.nn as nn
from utils import down_scale
from .residual import ResidualBlock

class DDANet(nn.Module):

  def __init__(self, in_channels, hid_channels, kernel_size, sam_groups, attention_size):
    super(DDANet, self).__init__()

    padding = kernel_size // 2

    self.conv_in = nn.Conv2d(in_channels, hid_channels, kernel_size, padding=padding)
    self.block1 = nn.Sequential(
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
    )
    
    self.down1 = nn.Conv2d(hid_channels, hid_channels*2, kernel_size, stride=2, padding=padding)
    self.input1 = nn.Conv2d(in_channels, hid_channels*2, kernel_size, padding=padding)
    self.project1 = nn.Conv2d(hid_channels*4, hid_channels*2, kernel_size, padding=padding)

    self.block2 = nn.Sequential(
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
    )
    self.down2 = nn.Conv2d(hid_channels*2, hid_channels*4, kernel_size, stride=2, padding=padding)
    self.input2 = nn.Conv2d(in_channels, hid_channels*4, kernel_size, padding=padding)
    self.project2 = nn.Conv2d(hid_channels*8, hid_channels*4, kernel_size, padding=padding)

    self.block3 = nn.Sequential(
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
    )
    self.block4 = nn.Sequential(
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(4*hid_channels, kernel_size, use_sam=True, use_fam_local=True, use_fam_global=True),
    )
    
    self.up1 = nn.ConvTranspose2d(hid_channels*4, hid_channels*2, kernel_size, stride=2, padding=padding, output_padding=padding)
    self.block5 = nn.Sequential(
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(2*hid_channels, kernel_size, use_sam=True, use_fam_local=True, use_fam_global=True),
    )

    self.up2 = nn.ConvTranspose2d(hid_channels*2, hid_channels, kernel_size, stride=2, padding=padding, output_padding=padding)
    self.block6 = nn.Sequential(
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size),
        ResidualBlock(hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_fam_local=True, use_fam_global=True),
        ResidualBlock(hid_channels, kernel_size, use_sam=True, use_fam_local=True, use_fam_global=True),
    )
    self.conv_out1 = nn.Conv2d(hid_channels, in_channels, kernel_size, padding=padding)
    self.conv_out2 = nn.Conv2d(hid_channels*2, in_channels, kernel_size, padding=padding)
    self.conv_out3 = nn.Conv2d(hid_channels*4, in_channels, kernel_size, padding=padding)

  def forward(self, X):
    X_2, X_4 = down_scale(X)

    # Encoder
    skip1 = self.block1(self.conv_in(X))
    input1 = self.project1(torch.concat((self.input1(X_2), self.down1(skip1)), dim=1))
    skip2 = self.block2(input1)

    input2 = self.project2(torch.concat((self.input2(X_4), self.down2(skip2)), dim=1))
    embedding = self.block3(input2)

    # Decoder
    up_out1 = self.block5(self.up1(self.block4(embedding)) + skip2)
    up_out2 = self.block6(self.up2(up_out1) + skip1)

    return (
        self.conv_out1(up_out2) + X,
        self.conv_out2(up_out1) + X_2,
        self.conv_out3(embedding) + X_4
    )