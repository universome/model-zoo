import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable


class ConditionalLM(nn.Module):
    """
    Two-layer GRU LM with binary one-hot feature
    concatenated to inputs of the second layer
    """
    def __init__(self, size, vocab):
        super(ConditionalLM, self).__init__()

        self.embed = nn.Embedding(len(vocab), size, padding_idx=vocab.stoi['<pad>'])
        self.grus = nn.ModuleList(nn.GRU(size, size, batch_first=True) for _ in range(2))
        self.interlayer = nn.Linear(size, size-1)
        self.out = nn.Linear(size, len(vocab))

    def forward(self, z, x, style, return_z=False):
        styles = cudable(torch.ones(x.size(0), x.size(1), 1) * style)
        x = self.embed(x)
        states, z_1 = self.grus[0](x, z[0].unsqueeze(0))
        states = self.interlayer(states)

        states = torch.cat([states, styles], dim=2)
        states, z_2 = self.grus[1](x, z[1].unsqueeze(0))
        out = self.out(states)

        if return_z:
            return out, torch.stack([z_1.squeeze(0), z_2.squeeze(0)])
        else:
            return out
