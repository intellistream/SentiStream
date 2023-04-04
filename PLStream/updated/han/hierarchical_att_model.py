import torch
import torch.nn as nn

from sent_att_model import SentAttNet
from word_att_model import WordAttNet


class HAN(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes,
                 pretrained_word_vec_path, max_sent_length, max_word_length):
        super().__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(
            pretrained_word_vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(
            sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        batch_size = self.batch_size

        if last_batch_size:
            batch_size = last_batch_size

        self.word_hidden_state = torch.zeros(
            2, batch_size, self.word_hidden_size, device='cuda')
        self.sent_hidden_state = torch.zeros(
            2, batch_size, self.sent_hidden_size, device='cuda')

    def forward(self, input_v):

        output_list = []
        input_v = input_v.permute(1, 0, 2)
        for i in input_v:
            output, self.word_hidden_state = self.word_att_net(
                i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(
            output, self.sent_hidden_state)

        return output
