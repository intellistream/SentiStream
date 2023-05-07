# pylint: disable=import-error
import torch.nn as nn


class Classifier(nn.Module):
    """
    Simple feedforward neural network for binary classification.

    Attributes:
        fc1 (nn.Linear): First fully connected layer of ANN.
        relu1 (nn.ReLU): ReLU activation function applied to output of first/input layer.
        fc2 (nn.Linear): Hidden fully connected layer of ANN.
        sigmoid (nn.Sigmoid): Sigmoid activation function applied to output/hidden layer.
    """

    def __init__(self, input_size, hidden_size=32, output_size=1):
        """
        Initialize class with given input and output dimensions.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer. Defaults to 32.
            output_size (int, optional): Number of output neurons (for binary 
                                        classification - using sigmoid) Defaults to 1.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Compute the forward pass of neural network on the given input.

        Args:
            x (torch.Tensor): Input tensor, with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor, with shape (batch_size, output_size).
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
