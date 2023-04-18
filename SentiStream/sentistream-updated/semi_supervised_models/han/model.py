# pylint: disable=import-error
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from semi_supervised_models.han.utils import mat_mul, element_wise_mul


class WordAttentionNet(nn.Module):
    """
    Word attention network of HAN.

    Attributes:
        word_bias (torch.nn.Parameter): Learnable bias for word-level attention.
        word_weight (torch.nn.Parameter): Learnable weight for word-level attention.
        context_weight (torch.nn.Parameter): Learnable weight for context-level attention.
        lookup (torch.nn.Embedding): Embedding layer for input sequence.
        gru (torch.nn.GRU): Word-level Bidirectional GRU layer.
    """

    def __init__(self, embeddings, hidden_size=50):
        """
        Initialize network with embeddings.

        Args:
            embeddings (ndarray): Pre-trained word embeddings.
            hidden_size (int, optional): Number of features in the hidden state of word-level GRU.
                                        Defaults to 50.
        """
        super().__init__()

        # Add row of zeros to the embeddings for padding and unknown words.
        pad_unk_word = np.zeros((1, embeddings.shape[1]))
        embeddings = torch.from_numpy(np.concatenate([pad_unk_word, embeddings], axis=0)
                                      .astype(np.float32))

        # Initialize learnable parameters.
        self.word_bias = nn.Parameter(torch.zeros(1, 2 * hidden_size))
        self.word_weight = nn.Parameter(torch.empty(
            2 * hidden_size, 2 * hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(
            torch.empty(2 * hidden_size, 1).normal_(0.0, 0.05))

        # Initialize embedding and GRU layers.
        self.lookup = nn.Embedding.from_pretrained(embeddings)

        self.gru = nn.GRU(embeddings.shape[1], hidden_size, bidirectional=True)

    def forward(self, x, hidden_state):
        """
        Compute forward pass of word attention network.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_length).
            hidden_state (torch.Tensor): Hidden state of GRU layer.

        Returns:
            torch.Tensor: Word attention-weighted representation of input sequence.
            torch.Tensor: Hidden state of GRU layer.
        """
        # Look up embeddings for input sequence.
        output = self.lookup(x)

        # Pass output through GRU layer.
        f_output, h_output = self.gru(output, hidden_state)
        self.gru.flatten_parameters()

        # Compute word-level attention and apply it to output of GRU layer.
        output = mat_mul(f_output, self.word_weight, self.word_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0))

        return output, h_output


class SentenceAttentionNet(nn.Module):
    """
    Sentence Attention Network of HAN.

    Attributes:
        sent_bias (Torch.nn.Parameter): Learnable bias for sentence-level attention.
        sent_weight (Torch.nn.Parameter): Learnable weight for sentence-level attention
        context_weight (Torch.nn.Parameter): Learnable weight for context-level attention
        gru (Torch.nn.GRU): Sentence-level Bidirectional GRU layer
        fc_out (Torch.nn.Linear): Fully connected output layer

    """

    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        """
        Intialize network

        Args:
            sent_hidden_size (int, optional): Number of features in hidden state of sentence-level 
                                            GRU. Defaults to 50.
            word_hidden_size (int, optional): Number of features in hidden state of word-level GRU.
                                            Defaults to 50.
        """
        super().__init__()

        # Initialize learnable parameters.
        self.sent_bias = nn.Parameter(
            torch.zeros(1, 2 * sent_hidden_size))
        self.sent_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 2 * sent_hidden_size).normal_(0.0, 0.05))
        self.context_weight = nn.Parameter(torch.empty(
            2 * sent_hidden_size, 1).normal_(0.0, 0.05))

        # Initialize GRU and Fully connected layers.
        self.gru = nn.GRU(2 * word_hidden_size,
                          sent_hidden_size, bidirectional=True)
        self.fc_out = nn.Linear(2 * sent_hidden_size, 1)

    def forward(self, x, hidden_state):
        """
        Compute forward pass of sentence attention network.

        Args:
            x (Torch.Tensor): Input sequence of shape (seq_len, batch_size, 2 * word_hidden_size).
            hidden_state (Torch.Tensor): Hidden state of GRU layer.

        Returns:
            Torch.Tensor: Sentence attention-weighted representation of input sequence.
            Torch.Tensor: Hidden state of GRU layer.

        """
        # Pass input through GRU layer.
        f_output, h_output = self.gru(x, hidden_state)
        self.gru.flatten_parameters()

        # Compute sentence-level attention and apply it to output of GRU layer.
        output = mat_mul(f_output, self.sent_weight, self.sent_bias)
        output = mat_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)

        # Fully connected output layer
        output = self.fc_out(output)

        return output, h_output


class HAN(nn.Module):
    """
    Hierarchical Attention Network

    Attributes:
        batch_size (int): Batch size.
        word_hidden_size (int): Size of word-level hidden state.
        sent_hidden_size (int): Size of sentence-level hidden state.
        max_sent_length (int): Maximum sentence length.
        max_word_length (int): Maximum word length.
        word_attention_net (WordAttentionNet): Word attention network.
        sentence_attention_net (SentenceAttentionNet): Sentence attention network.
        device (torch.device): Device on which to run the model.
    """

    def __init__(self, embeddings, batch_size=128, max_sent_length=10, max_word_length=15,
                 word_hidden_size=50, sent_hidden_size=50):
        """
        Initialize HAN.

        Args:
            word_hidden_size (int): Hidden state size for word-level attention layer.Defaults to 50.
            sent_hidden_size (int): Hidden state size for sentence-level attention layer. Defaults
                                    to 50.
            batch_size (int): Batch size. Defaults to 128.
            embeddings (torch.Tensor): Pre-trained embeddings.
            max_sent_length (int): Maximum sentence length. Defaults to 10.
            max_word_length (int): Maximum word length. Defaults to 15.
        """
        super().__init__()

        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length

        # Initialize word-level attention network.
        self.word_attention_net = WordAttentionNet(
            embeddings, word_hidden_size)

        # Initialize sentence-level attention network.
        self.sentence_attention_net = SentenceAttentionNet(
            sent_hidden_size, word_hidden_size)
        
        # Output layer
        self.sigmoid = nn.Sigmoid()

        # Set device to use for computations.
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Reset hidden state for word-level and sentence-level GRUs.
        self.reset_hidden_state()

    def reset_hidden_state(self, last_batch_size=None):
        """
        Reset hidden state of model.

        Args:
            last_batch_size (int, optional): Last batch's size. Defaults to None.
        """
        batch_size = self.batch_size

        if last_batch_size:
            batch_size = last_batch_size

        # Reset word-level hidden state.
        self.word_hidden_state = torch.zeros(
            2, batch_size, self.word_hidden_size, device=self.device)

        # Reset the sentence-level hidden state.
        self.sent_hidden_state = torch.zeros(
            2, batch_size, self.sent_hidden_size, device=self.device)

    def forward(self, input_vector):
        """
        Compute forward pass of HAN model.

        Args:
            input_vector (torch.Tensor): Input sequence of shape (batch_size, max_sent_length,
                                        max_word_length).

        Returns:
            torch.Tensor: Output sequence of model of shape (batch_size, num_classes)
        """
        # Permute input tensor to match the expected shape of the word-level attention network.
        input_vector = input_vector.permute(1, 0, 2)

        # Compute the word-level attention weights for each sentence in the input tensor.
        output_list = [self.word_attention_net(
            sentence.permute(1, 0), self.word_hidden_state)[0] for sentence in input_vector]

        # Concatenate output tensors for each sentence to form tensor of shape
        # (batch_size * max_sent_length, 2 * word_hidden_size)
        output = torch.cat(output_list, 0)

        # Compute sentence-level attention weights for concatenated output tensor
        # pylint: disable=attribute-defined-outside-init
        # Since reset_hidden_state should be updated for each batch, defining its var in init may
        # cause unwanted computations
        output, self.sent_hidden_state = self.sentence_attention_net(
            output, self.sent_hidden_state)

        output = self.sigmoid(output)

        return output
