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

        masks = create_masks(input_size, channels)
        self.layers = nn.ModuleList([RevLayer(m) for m in masks])

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


def create_masks(input_size:int, channels:List[int]):
    masks = []
    current_size = input_size

    for i, (prev_c, curr_c) in enumerate(zip(channels[:-1], channels[1:])):
        if prev_c < curr_c:
            current_size = int(current_size // 2)
        elif prev_c > curr_c:
            current_size = int(current_size * 2)

        if i % 2 == 0:
            mask = create_checkerboard_mask(current_size, zero_diag=True)
        else:
            mask = create_checkerboard_mask(current_size, zero_diag=False)

        mask = mask.unsqueeze(0).repeat(curr_c, 1, 1)
        masks.append(mask.float())

    return masks