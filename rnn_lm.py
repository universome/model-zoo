import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable


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

    def inference(self, x, vocab, max_len=100, eos_token='<eos>'):
        "Continue sequence given a subsequence"
        assert x.dim() == 1 # batch_size is 1 and we have only list of token_idx

        result = []
        x = self.embed(x.unsqueeze(0))
        z = self.gru(x)[1]
        current_token = vocab.stoi[eos_token]

        for _ in range(max_len):
            token = cudable(torch.tensor([[current_token]]))
            emb = self.embed(token)
            z = self.gru(emb, z)[1]
            next_token = self.out(z[:, -1]).max(dim=-1)[1].item()

            result.append(next_token)
            current_token = next_token

            if next_token == vocab.stoi[eos_token]: break

        return result
