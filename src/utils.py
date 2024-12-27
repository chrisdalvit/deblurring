import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torchvision.transforms import v2

def down_scale(x):
  x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
  x_4 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear')
  return x_2, x_4

def loss_spatial(outputs, x):
  criterion = nn.L1Loss(reduction='mean')
  x_2, x_4 = down_scale(x)
  l1 = criterion(outputs[0], x)
  l2 = criterion(outputs[1], x_2)
  l3 = criterion(outputs[2], x_4)
  return l1 + l2 + l3

def loss_freq(outputs, x):
  criterion = nn.L1Loss(reduction='mean')
  x_2, x_4 = down_scale(x)
  outputs_fft = [ fft.rfft(o) for o in outputs ]
  x_fft = fft.rfft(x)
  x_2_fft = fft.rfft(x_2)
  x_4_fft = fft.rfft(x_4)
  l1 = criterion(outputs_fft[0], x_fft)
  l2 = criterion(outputs_fft[1], x_2_fft)
  l3 = criterion(outputs_fft[2], x_4_fft)
  return l1 + l2 + l3

def loss_fn(outputs, gt, lam=0.1):
  return loss_spatial(outputs, gt) + lam*loss_freq(outputs, gt)

def get_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(size=(256, 256), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])