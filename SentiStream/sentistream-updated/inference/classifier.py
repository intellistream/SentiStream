# pylint: disable=import-error
# pylint: disable=no-name-in-module

import time
import numpy as np
import torch

from sklearn.metrics import accuracy_score

import config

from semi_supervised_models.ann.model import Classifier as ANN
from semi_supervised_models.han.model import HAN

from semi_supervised_models.han.utils import join_tokens, preprocess
from utils import load_torch_model, get_average_word_embeddings, clean_for_wv


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
        start_time (float): Start time of classification.

    Constants:
        TIME_TO_UPDATE: Time interval in seconds to load updated word vector and/or NN model.
    """
    TIME_TO_UPDATE = 600

    def __init__(self, word_vector_algo, ssl_model, batch_size=None, is_eval=False):
        """
        Initialize class with pretrained models.

        Args:
            word_vector_algo (class): Type of word vector algorithm to use (either 'Word2Vec' or
                                    'FastText').
            ssl_model (str): Type of SSL model to use (either 'ANN' or 'HAN').
            batch_size (_type_, optional): Batch size to use for processing data. Defaults to None.
            is_eval (bool, optional): Flag indicating whether to use model for prediction or
                                    evaluation. Defaults to False.
        """

        # Determine if GPU available for inference.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load models.
        self.wv_model = word_vector_algo.load(config.SSL_WV)
        self.clf_model = load_torch_model(
            ANN(self.wv_model.vector_size) if ssl_model == 'ANN' else HAN(np.array([
                self.wv_model.wv[key] for key in self.wv_model.wv.index_to_key])),
            config.SSL_CLF).to(self.device)
        self.ssl_model = ssl_model

        self.acc_list = []

        # Set classifier mode. If evaluation, model predicts labels for stream data else model used
        # to test accuracy on pseudo labels
        self.is_eval = is_eval

        # Set batch size and initialize lists for labels and texts.
        self.batch_size = batch_size if batch_size is not None else (
            16 if ssl_model == 'ANN' else 128)

        self.idx = []
        self.labels = []
        self.texts = []

        # Set start time of classifier.
        self.start_time = time.time()

    def get_prediction(self, data):
        """
        Get NN model's prediction for batch data.

        Args:
            data (ndarray): Word embeddings to input to model for prediction.

        Returns:
            tuple: Predictions and it's confidence scores for current batch.
        """

        with torch.no_grad():
            if self.ssl_model == 'ANN':
                # Get predicted probabilities.
                preds = self.clf_model(torch.FloatTensor(data).to(self.device))

                # Calculate binary predictions and confidence scores.
                conf = (torch.abs(preds - 0.5) * 2).view(-1)
                preds = preds.ge(0.5).long().view(-1)

            else:
                # Reset hidden state for current batch.
                self.clf_model.reset_hidden_state(data.shape[0])
                # Get predicted probabilities.
                preds = self.clf_model(torch.from_numpy(data).to(self.device))

                # Calculate binary predictions and confidence scores.
                max_t, _ = torch.max(preds, 1)
                min_t, _ = torch.min(preds, 1)
                conf = torch.abs(max_t / (max_t - min_t))
                _, preds = torch.max(preds, 1)

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
        idx, label, text = data

        self.idx.append(idx)
        self.labels.append(label)
        self.texts.append(text)

        # Check if batch size is reached.
        if len(self.labels) >= self.batch_size:

            # Get document embeddings.
            if self.ssl_model == 'ANN':
                embeddings = get_average_word_embeddings(
                    self.wv_model, [clean_for_wv(tokens) for tokens in self.texts])
            else:
                embeddings = preprocess(join_tokens(
                    self.texts), self.wv_model.wv.key_to_index)

            # Get predictions and confidence scores.
            conf, preds = self.get_prediction(
                np.array(embeddings))

            # Calculate model's accuracy.
            output = accuracy_score(self.labels, preds)
            self.acc_list.append(output)

            if not self.is_eval:
                # Generate output data.
                output = [[i, 'ss', c, p, t]
                          for i, c, p, t in zip(self.idx, conf, preds, self.texts)]

            # Clear the lists for the next batch.
            self.idx = []
            self.labels = []
            self.texts = []

            return output
        return config.BATCHING
