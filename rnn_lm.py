import torch
import torch.nn as nn


class RNNLM(nn.Module):
    def __init__(self, size, vocab, n_layers=1):
        super(RNNLM, self).__init__()

        self.embed = nn.Embedding(len(vocab), size, padding_idx=vocab.stoi['<pad>'])
        self.gru = nn.GRU(size, size, batch_first=True, num_layers=n_layers)
        self.out = nn.Linear(size, len(vocab))

    def forward(self, z, x, return_z=False):
        x = self.embed(x)
        states, z = self.gru(x, z)
        out = self.out(states)

        return (out, z) if return_z else out
