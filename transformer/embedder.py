import torch
import torch.nn as nn
import numpy as np
from firelab.utils.training_utils import cudable

from .layers import PositionalEncoding
from .decoder import DecoderLayer


class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()

        self.config = config
        self.layer = DecoderLayer(config)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.embed = nn.Embedding(self.config.n_vecs, self.config.d_model)

    def forward(self, encs, encs_mask):
        tokens = torch.arange(self.config.n_vecs).long().unsqueeze(0).repeat(encs.size(0), 1)
        x = self.embed(cudable(tokens))
        x = x * np.sqrt(self.config.d_model)

        for _ in range(self.config.n_steps):
            x = self.layer(encs, x, encs_mask, None)

        x = self.norm(x)

        return x
