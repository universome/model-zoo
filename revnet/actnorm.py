import torch
import torch.nn as nn


class ActNorm(nn.Module):
    """
    ActNorm layer from Glow.
    Taken from https://github.com/rosinality/glow-pytorch
    """
    def __init__(self, in_channels):
        super(ActNorm, self).__init__()

        self.initialized = False
        self.shift = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def initialize(self, x, eps=1e-8):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)

            x_mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            x_std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.shift.data.copy_(x_mean)
            self.scale.data.copy_(x_std + eps)

        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        return (x - self.shift) / (self.scale + 1e-8)

    def reverse_forward(self, y):
        return y * self.scale + self.shift
