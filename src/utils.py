import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torchvision.transforms import v2

def down_scale(x):
  x_2 = F.interpolate(x, scale_factor=0.5)
  x_4 = F.interpolate(x_2, scale_factor=0.5)
  return x_2, x_4

def loss_spatial(outputs, gt):
  l1 = nn.L1Loss(reduction='mean')
  gt_scales = (gt, *down_scale(gt))
  return sum(l1(x, y) for x, y in zip(outputs, gt_scales))

def loss_freq(outputs, gt):
  l1 = nn.L1Loss(reduction='mean')
  gt_scales = (gt, *down_scale(gt))
  return sum(l1(fft.rfft(x), fft.rfft(y)) for x, y in zip(outputs, gt_scales))

def loss_fn(outputs, gt, lam=0.1):
  return loss_spatial(outputs, gt) + lam*loss_freq(outputs, gt)

def get_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(size=(256, 256), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])