import random

import torch
import numpy as np


def create_random_mask(size:int, n_filled:int) -> np.ndarray:
    mask = [True] * n_filled + [False] * (size - n_filled)
    mask = random.sample(mask, len(mask))
    mask = np.array(mask)

    return mask


def create_checkerboard_mask(size:int, zero_diag:bool=True):
    pattern = [[0, 1], [1, 0]] if zero_diag else [[1,0], [0, 1]]
    mask = np.tile(pattern, (size // 2 + 1, size // 2 + 1)).astype(np.uint8)
    mask = mask[:size, :size]
    mask = torch.tensor(mask, requires_grad=False)

    return mask
