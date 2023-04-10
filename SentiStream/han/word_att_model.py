import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from utils import mat_mul, ele_wise_mul


class WordAttNet(nn.Module):
    def __init__(self, embeddings, hidden_size=50):
        super().__init__()
        pad_unk_word = np.zeros((1, embeddings.shape[1]))
        embeddings = torch.from_numpy(np.concatenate([pad_unk_word, embeddings], axis=0).astype(
            np.float32))  # try with float16 - mixedprecision and check metrics ##################

        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size).zero_())
        self.word_weight = nn.Parameter(torch.empty(
            2 * hidden_size, 2 * hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(
            torch.empty(2 * hidden_size, 1).normal_(0.0, 0.05))

        self.lookup = nn.Embedding.from_pretrained(embeddings)
        self.gru = nn.GRU(embeddings.shape[1], hidden_size, bidirectional=True)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output, hidden_state)

        output = mat_mul(f_output, self.word_weight, self.word_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = ele_wise_mul(f_output, output.permute(1, 0))

        return output, h_output
