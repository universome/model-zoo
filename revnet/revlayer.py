import torch
import torch.nn as nn
from firelab.config import Config

from .actnorm import ActNorm


class RevLayer(nn.Module):
    def __init__(self, mask:torch.Tensor):
        super(RevLayer, self).__init__()

        self.mask = mask
        self.actnorm = ActNorm(self.mask.size(0))
        self.mult = ConvBlock(self.mask.size(0))
        self.bias = ConvBlock(self.mask.size(0))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)

        out = x.view(x.size(0), *self.mask.size())
        out = self.actnorm(out)
        changing, same = out * self.mask, out * (1 - self.mask)
        changing = (self.mult(same) + 2).sigmoid() * changing + self.bias(same)
        out = changing * self.mask + same

        return out

    def reverse_forward(self, y:torch.Tensor) -> torch.Tensor:
        if self.mask.device != y.device:
            self.mask = self.mask.to(y.device)

        out = y.view(y.size(0), *self.mask.size())
        changing, same = out * self.mask, out * (1 - self.mask)
        changing = (changing - self.bias(same)) / (self.mult(same) + 2).sigmoid()
        out = changing * self.mask + same
        out = self.actnorm.reverse_forward(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels:int):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out
