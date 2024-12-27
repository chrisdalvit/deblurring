import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):

  def __init__(self, channels, groups, attention_size, kernel_size=3):
    super(SAM, self).__init__()
    self.groups = groups
    self.attention_size = attention_size
    self.padding = kernel_size // 2
    self.conv1 = nn.Conv2d(channels, groups * attention_size**2, kernel_size=1)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=self.padding)
    self.tanh = nn.Tanh()

  def forward(self, input):
    # input (N,C,H,W)
    batch, channels, height, width = input.size()
    assert channels % self.groups == 0, f"Assert channel size {channels} is divisible by group size {self.groups}"
    means = input.mean(dim=(2,3), keepdim=True) # Apply global average pooling
    W = self.tanh(self.conv1(means)).view(batch, self.groups, self.attention_size, self.attention_size)

    group_convs = []
    group_size = channels // self.groups
    for Xi, Wi in zip(input, W):
        kernels = torch.cat([kernel.repeat(group_size, group_size, 1, 1) for kernel in Wi], dim=0)  # Precompute all kernels
        attention = F.conv2d(Xi, kernels, padding=1, groups=self.groups)  # Batched convolution
        group_convs.append(attention)

    S = torch.cat(group_convs, dim=0).view_as(input)
    return self.conv2(S)