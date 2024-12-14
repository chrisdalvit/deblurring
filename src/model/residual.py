import torch.nn as nn
from .sam import SAM
from .fam import FAM

class ResidualBlock(nn.Module):

  def __init__(self, channels, kernel_size):
    super(ResidualBlock, self).__init__()

    padding = kernel_size // 2
    self.main = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU(),
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU()
    )

  def forward(self, X):
    return self.main(X) + X

class ResidualAttentionBlock(nn.Module):

  def __init__(self, channels, kernel_size, groups, attention_size):
    super(ResidualAttentionBlock, self).__init__()

    padding = kernel_size // 2
    self.main = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU(),
      SAM(channels, groups, attention_size),
      FAM(channels),
      nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
      nn.ReLU()
    )

  def forward(self, X):
    return self.main(X) + X

class CombinedResidalBlock(nn.Module):

  def __init__(self, channels, groups, attention_size, kernel_size) -> None:
    super(CombinedResidalBlock, self).__init__()
    self.main = nn.Sequential(
        ResidualBlock(channels, kernel_size),
        ResidualAttentionBlock(channels, kernel_size, groups, attention_size)
    )

  def forward(self, X):
    return self.main(X)