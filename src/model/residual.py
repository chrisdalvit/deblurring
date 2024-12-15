import torch.nn as nn
from .sam import SAM
from .fam import FAM

class ResidualBlock(nn.Module):

  def __init__(self, channels, kernel_size, use_sam=False, use_fam_local=False, use_fam_global=False, groups=8, attention_size=3):
    super(ResidualBlock, self).__init__()

    padding = kernel_size // 2
    modules = [
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU(),        
    ]
    if use_sam:
        modules.append(SAM(channels, groups, attention_size))
    if use_fam_local or use_fam_global:
        modules.append(FAM(channels, local=use_fam_local, glob=use_fam_global))
    modules.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding))
    modules.append(nn.ReLU())
    self.main = nn.Sequential(*modules)

  def forward(self, X):
    return self.main(X) + X
