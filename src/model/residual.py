import torch.nn as nn
from .sam import SAM
from .fam import FAM

class ResidualBlock(nn.Module):

  def __init__(self, channels, kernel_size, use_sam=False, fam_mode='all', groups=8, attention_size=3):
    super(ResidualBlock, self).__init__()

    padding = kernel_size // 2
    modules = [
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU(),        
    ]
    if use_sam:
        modules.append(SAM(channels, groups, attention_size))
    if fam_mode != 'none':
        modules.append(FAM(channels, mode=fam_mode))
    modules.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding))
    modules.append(nn.ReLU())
    self.main = nn.Sequential(*modules)

  def forward(self, X):
    return self.main(X) + X
