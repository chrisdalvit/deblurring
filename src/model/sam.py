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
    for Xi, Wi in zip(input, W):
      group_size = channels // self.groups
      for idx, kernel in enumerate(Wi):
        group = Xi[idx*group_size:idx*group_size+group_size,:]
        y = kernel.repeat(group_size, group_size, 1, 1)
        attention = F.conv2d(group, y, padding=1)
        group_convs.append(attention)
    S = torch.cat(group_convs).view_as(input)
    return self.conv2(S)