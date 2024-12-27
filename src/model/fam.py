import torch
import torch.nn as nn

class FAM(nn.Module):

  def __init__(self, channels, mode='all', kernel_size=3):
    super(FAM, self).__init__()
    padding = kernel_size // 2
    self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding)
    self.W_gl = nn.Parameter(torch.zeros(channels, 1, 1))
    self.W_gh = nn.Parameter(torch.zeros(channels, 1, 1))
    self.W_ll = nn.Parameter(torch.zeros(channels, 1, 1))
    self.W_lh = nn.Parameter(torch.zeros(channels, 1, 1))
    self.mode = mode

  def forward(self, X):
    X_gl = X.mean(dim=(2,3), keepdim=True).expand(X.size())
    X_gh = X - X_gl
    X_ll = self.avg_pool(X)
    X_lh = X - X_ll
    if self.mode == 'all':
      return self.W_gl * X_gl + self.W_gh * X_gh + self.W_ll * X_ll + self.W_lh * X_lh
    elif self.mode == 'global':
      return self.W_gl * X_gl + self.W_gh * X_gh 
    elif self.mode == 'local':
      return self.W_ll * X_ll + self.W_lh * X_lh