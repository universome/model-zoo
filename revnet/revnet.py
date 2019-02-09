from typing import List

import torch
import torch.nn as nn
import numpy as np

from .revlayer import RevLayer
from .utils import create_checkerboard_mask


class RevNet(nn.Module):
    """
    Reversible NN. Consider using double type everywhere
    if you want precise inversion
    """
    def __init__(self, input_size:int, channels:List[int]):
        super(RevNet, self).__init__()

        masks = self.create_masks(input_size, channels)
        self.layers = nn.ModuleList([RevLayer(m) for m in masks])

    def create_masks(self, input_size:int, channels:List[int]):
        masks = []

        for i, c in enumerate(channels):
            current_size = int(input_size // np.sqrt(c))

            if i % 2 == 0:
                mask = create_checkerboard_mask(current_size, zero_diag=True)
            else:
                mask = create_checkerboard_mask(current_size, zero_diag=False)

            mask = mask.unsqueeze(0).repeat(c, 1, 1)
            masks.append(mask.float())

        return masks

    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer(out)

        return out

    def reverse_forward(self, y):
        out = y

        for layer in self.layers[::-1]:
            out = layer.reverse_forward(out)

        return out
