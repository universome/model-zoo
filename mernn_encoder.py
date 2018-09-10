from typing import List

import torch
import torch.nn as nn
from firelab.utils.training_utils import cudable


class MERNNEncoder(nn.Module):
    "RNN encoder which outputs mutiple embeddings with a variant of ACT mechanism"
    def __init__(self, size, vocab, output_threshold=0.95):
        super(MERNNEncoder, self).__init__()

        self.output_threshold = output_threshold
        self.embeddings = nn.Embedding(len(vocab), size, padding_idx=vocab.stoi['<pad>'])
        self.gru = nn.GRU(size, size, batch_first=True)
        self.act = nn.Sequential(nn.Linear(size, 1), nn.Sigmoid())

    def forward(self, sentence):
        self.gru.flatten_parameters()

        word_embs = self.embeddings(sentence)
        states = self.gru(word_embs)[0].squeeze(0)
        output_probs = self.act(states)
        embs_list = self.weighted_sum(states, output_probs)
        embs = self.pad(embs_list)

        return embs, output_probs

    def weighted_sum(self, states, probs):
        assert states.size(0) == probs.size(0)
        assert states.size(1) == probs.size(1)
        assert states.dim() == 3
        assert probs.dim() == 2

        batch_size = states.size(0)
        result = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            halt_sum = 0
            n_steps = 0

            for j in range(states.size(1)):
                halt_sum += probs[i][j].item()
                n_steps += 1

                if halt_sum > self.output_threshold:
                    emb = self.build_emb(states[i][j - (n_steps-1)], probs[i][j - (n_steps-1)])
                    result[i].append(emb)
                    n_steps = 0
                    halt_sum = 0

            result[i] = torch.cat(result[i])

        return result

    def build_emb(self, states, probs):
        assert states.size(0) == probs.size(0)
        assert states.dim() == 2
        assert probs.dim() == 1

        last_prob = probs[-1] - (probs.sum() - 1).detach()
        emb = states[:-1] * probs[:-1].unsqueeze(1) + states[-1] + last_prob

        return emb

    def pad(self, seqs: List[torch.Tensor]):
        "Pads sequences with zero vectors and concats them"
        longest = max(len(s) for s in seqs)
        seqs = [self.pad_to(s, longest) for s in seqs]

        return torch.cat(seqs)

    def pad_to(self, seq, n):
        "Pads sequence with zero vectors to the desired length"
        assert len(seq) <= n

        if len(seq) == n: return seq

        pads = torch.zeros(n - len(seq), seq.size(1))
        pads = cudable(pads).type(seq.type())

        return torch.cat([seq, pads])

