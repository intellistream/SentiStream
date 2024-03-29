# pylint: disable=import-error
# pylint: disable=no-name-in-module
import numpy as np
import torch

from gensim.models import Word2Vec

import config

from semi_supervised_models.utils import join_tokens, preprocess
from semi_supervised_models.model import HAN
from utils import load_torch_model


class Classifier:
    """
    Classify polarity using trained word vector and NN model.

    Attributes:
        wv_model (class): Pre-trained word vector model.
        clf_model (class): Pre-trained PyTorch model.
        ssl_model (str): Type of SSL model used for classification (either 'ANN' or 'HAN').
        acc_list (list): Store accuracy of each batch.
        batch_size (int): Batch size to use for processing data.
        labels (list): Labels for each text in the current batch.
        texts (list): Cleaned and tokenized texts in the current batch.
    """

    def __init__(self, batch_size=10000):
        """
        Initialize class with pretrained models.

        Args:
            batch_size (_type_, optional): Batch size to use for processing data. Defaults to 10000.
        """
        # Determine if GPU available for inference.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.eval_list = []

        self.wv_model, self.clf_model = None, None

        # Set batch size and initialize lists for labels and texts.
        self.batch_size = batch_size

        self.id = []
        self.idx = []
        self.labels = []
        self.texts = []

    def load_updated_model(self):
        """
        Load updated model from local.
        """
        self.wv_model = Word2Vec.load(config.SSL_WV)
        self.clf_model = load_torch_model(HAN(np.array(
            [self.wv_model.wv[key] for key in self.wv_model.wv.index_to_key]),
            batch_size=self.batch_size), config.SSL_CLF).to(self.device)

    def get_prediction(self, data):
        """
        Get NN model's prediction for batch data.

        Args:
            data (ndarray): Word embeddings to input to model for prediction.

        Returns:
            tuple: Predictions and it's confidence scores for current batch.
        """
        with torch.no_grad():
            # Get predicted probabilities.
            preds = self.clf_model(torch.from_numpy(data).to(self.device))

            # Calculate binary predictions and confidence scores.
            conf = (torch.abs(preds - 0.5) * 2).view(-1)
            preds = preds.ge(0.5).long().view(-1)

        return conf.tolist(), preds.tolist()

    def classify(self, data):
        """
        Classify incoming stream data by batch.

        Args:
            data (tuple): Contains index, label and text data.

        Returns:
            tuple or str: 'BATCHING' if collection data for batch, else, tuple containing ground 
                        truth label, confidence score and predicted label for current batch.
        """
        id, idx, label, text = data

        self.id.append(id)
        self.idx.append(idx)
        self.labels.append(label)
        self.texts.append(text)

        # Check if batch size is reached.
        if len(self.labels) >= self.batch_size or id == '-1':
            self.batch_size = len(self.labels)
            self.load_updated_model()

            embeddings = np.array(preprocess(join_tokens(
                self.texts), self.wv_model.wv.key_to_index))

            # Get predictions and confidence scores.
            conf, preds = self.get_prediction(embeddings)

            self.eval_list += list(zip(self.id, preds, self.labels))

            # Generate output data.
            output = [[i, 'ss', c, p, t]
                      for i, c, p, t in zip(self.idx, conf, preds, self.texts)]

            # Clear the lists for the next batch.
            self.id = []
            self.idx = []
            self.labels = []
            self.texts = []

            return output
