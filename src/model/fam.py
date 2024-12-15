import torch.nn as nn

class FAM(nn.Module):

  def __init__(self, channels, local, glob, kernel_size=3):
    super(FAM, self).__init__()
    padding = kernel_size // 2
    self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
    self.conv_gl = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
    self.conv_gh = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
    self.conv_ll = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
    self.conv_lh = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
    self.local = local
    self.glob = glob

  def forward(self, X):
    convs = []
    if self.glob:
      X_gl = X.mean(dim=(2,3), keepdim=True).expand(X.size())
      X_gh = X - X_gl
      convs.append(self.conv_gl(X_gl))
      convs.append(self.conv_gh(X_gh))
    if self.local:
      X_ll = self.avg_pool(X)
      X_lh = X - X_ll
      convs.append(self.conv_ll(X_ll))
      convs.append(self.conv_lh(X_lh))
    return sum(convs)