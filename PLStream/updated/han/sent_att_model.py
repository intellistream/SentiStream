import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mat_mul, ele_wise_mul


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2):
        super().__init__()

        self.sent_bias = nn.Parameter(
            torch.empty(1, 2 * sent_hidden_size).zero_())
        self.sent_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 2 * sent_hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 1).normal_(0.0, 0.05))

        self.gru = nn.GRU(2 * word_hidden_size,
                          sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = mat_mul(f_output, self.sent_weight, self.sent_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = ele_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output
