import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import mat_mul, ele_wise_mul
import pandas as pd
import numpy as np


class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(word2vec_path, header=None,
                           sep=" ", quoting=3).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        pad_unk_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([pad_unk_word, dict], axis=0).astype(
            np.float32))  # dont change order  ### try with float16 and check metrics

        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size).zero_())
        self.word_weight = nn.Parameter(torch.empty(
            2 * hidden_size, 2 * hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(
            torch.empty(2 * hidden_size, 1).normal_(0.0, 0.05))

        self.lookup = nn.Embedding.from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        # feature output and hidden state output
        f_output, h_output = self.gru(output, hidden_state)

        output = mat_mul(f_output, self.word_weight, self.word_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = ele_wise_mul(f_output, output.permute(1, 0))

        return output, h_output
