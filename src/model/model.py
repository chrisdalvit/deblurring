import torch
import torch.nn as nn
from utils import down_scale
from .residual import ResidualBlock
from .scm import SCM

class ScaleBlock(nn.Module):
    
    def __init__(self, n_channels, kernel_size):
        super(ScaleBlock, self).__init__()
        self.main = nn.Sequential(
            ResidualBlock(n_channels, kernel_size),
            ResidualBlock(n_channels, kernel_size, use_sam=True, fam_mode='all'),
        )
       
    def forward(self, x):
        return self.main(x)
    
    
class ProjectionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
       super(ProjectionBlock, self).__init__()
       self.main = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
           nn.ReLU(inplace=True)
       )
       
    def forward(self, x):
        return self.main(x)
    

class DDANet(nn.Module):

  def __init__(self, in_channels, hid_channels, kernel_size=3):
    super(DDANet, self).__init__()

    padding = kernel_size // 2

    self.conv_in = nn.Conv2d(in_channels, hid_channels, kernel_size, padding=padding)
    self.encoders = nn.ModuleList([
        ScaleBlock(hid_channels, kernel_size),
        ScaleBlock(2*hid_channels, kernel_size),
        ScaleBlock(4*hid_channels, kernel_size)
    ])
    
    self.decoders = nn.ModuleList([
        ScaleBlock(4*hid_channels, kernel_size),
        ScaleBlock(2*hid_channels, kernel_size),
        ScaleBlock(hid_channels, kernel_size)
    ])
    
    self.inputs = nn.ModuleList([
        SCM(hid_channels*2),
        SCM(hid_channels*4)        
    ])
    
    self.conv_outs = nn.ModuleList([
        nn.Conv2d(hid_channels, in_channels, kernel_size, padding=padding),
        nn.Conv2d(hid_channels*2, in_channels, kernel_size, padding=padding),
        nn.Conv2d(hid_channels*4, in_channels, kernel_size, padding=padding)
    ])
    
    self.down_sacles = nn.ModuleList([
        nn.Conv2d(hid_channels, hid_channels*2, kernel_size, stride=2, padding=padding),
        nn.Conv2d(hid_channels*2, hid_channels*4, kernel_size, stride=2, padding=padding)        
    ])
    
    self.up_scales = nn.ModuleList([
        nn.ConvTranspose2d(hid_channels*4, hid_channels*2, kernel_size, stride=2, padding=padding, output_padding=padding),
        nn.ConvTranspose2d(hid_channels*2, hid_channels, kernel_size, stride=2, padding=padding, output_padding=padding)  
    ])
    
    self.projections = nn.ModuleList([
        ProjectionBlock(hid_channels*2, hid_channels, kernel_size, padding=padding),
        ProjectionBlock(hid_channels*4, hid_channels*2, kernel_size, padding=padding),
        ProjectionBlock(hid_channels*8, hid_channels*4, kernel_size, padding=padding),
    ])

  def forward(self, x):
    x_2, x_4 = down_scale(x)
    z = self.conv_in(x)
    z_2 = self.inputs[0](x_2)
    z_4 = self.inputs[1](x_4)

    # Encoder
    skip1 = self.encoders[0](z)
    
    scale2 = self.down_sacles[0](skip1)
    scale2 = torch.concat([z_2, scale2], dim=1)
    scale2 = self.projections[1](scale2)
    skip2 = self.encoders[1](scale2)

    scale4 = self.down_sacles[1](skip2)
    scale4 = torch.concat([z_4, scale4], dim=1)
    scale4 = self.projections[2](scale4)
    out4 = self.encoders[2](scale4)

    # Decoder
    emb4 = self.decoders[0](out4)
    emb2 = torch.concat([self.up_scales[0](emb4), skip2], dim=1)
    emb2 = self.projections[1](emb2)
    out2 = self.decoders[1](emb2)
    
    emb = torch.concat([self.up_scales[1](out2), skip1], dim=1)
    emb = self.projections[0](emb)
    out = self.decoders[2](emb)

    return (
        self.conv_outs[0](out) + x,
        self.conv_outs[1](out2) + x_2,
        self.conv_outs[2](out4) + x_4
    )