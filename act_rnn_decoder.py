import torch.nn as nn


class ACTRNNDecoder(nn.Module):
    def __init__(self, size, vocab, halt_threshold=0.95):
        super(ACTRNNDecoder, self).__init__()

        self.embed = nn.Embedding(len(vocab), size, padding_idx=vocab.stoi['<pad>'])
        self.gru = nn.GRU(size, size, batch_first=True)
        self.z_to_logits = nn.Linear(size, len(vocab))

    def forward(self, states, sentences):
        self.gru.flatten_parameters()

        embs = self.embed(sentences)
        hid_states, _ = self.gru(embs, z.unsqueeze(0))
        logits = self.z_to_logits(hid_states)

        return logits
