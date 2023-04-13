# pylint: disable=import-error
# pylint: disable=no-name-in-module

import time
import numpy as np
import torch

from semi_supervised_models.han.utils import join_tokens, preprocess
from utils import (load_word_vector_model, load_torch_model,
                   clean_for_wv, tokenize, get_average_word_embeddings)


class Classifier:
    """
    Classify polarity using trained word vector and NN model.

    Attributes:
        wv_model: Pre-trained word vector model.
        clf_model: Pre-trained PyTorch model.
        ssl_model: Type of SSL model used for classification (either 'ANN' or 'HAN').
        batch_size: Batch size to use for processing data.
        labels: Labels for each text in the current batch.
        texts: Cleaned and tokenized texts in the current batch.
        start_time: Start time of classification.

    Constants:
        TIME_TO_UPDATE: Time interval in seconds to load updated word vector and/or NN model.
    """
    TIME_TO_UPDATE = 600

    def __init__(self, word_vector_algo, ssl_model, batch_size=None):
        """
        Initialize class with pretrained models.

        Args:
            word_vector_algo (class): Type of word vector algorithm to use (either 'Word2Vec' or
                                    'FastText').
            ssl_model (str): Type of SSL model to use (either 'ANN' or 'HAN').
            batch_size (_type_, optional): Batch size to use for processing data. Defaults to None.
        """

        # Determine if GPU available for training.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load models.
        self.wv_model = load_word_vector_model(
            word_vector_algo, 'ssl-wv.model')
        self.clf_model = load_torch_model('ssl-clf.pth').to(self.device)
        self.ssl_model = ssl_model

        # Set batch size and initialize lists for labels and texts.
        self.batch_size = batch_size if batch_size is not None else (
            16 if ssl_model == 'ANN' else 128)

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
                conf = torch.abs(preds - 0.5) * 2
                preds = preds.gt(0.5).long()

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
            data (tuple): Contains label and text data.

        Returns:
            tuple or str: 'BATCHING' if collection data for batch, else, tuple containing ground 
                        truth label, confidence score and predicted label for current batch.
        """
        label, text = data

        self.labels.append(label)
        self.texts.append(clean_for_wv(tokenize(text)))

        # Check if batch size is reached.
        if len(self.labels) >= self.batch_size:

            # Get document embeddings.
            if self.ssl_model == 'ANN':
                embeddings = [get_average_word_embeddings(
                    self.wv_model, tokens) for tokens in self.texts]
            else:
                embeddings = preprocess(join_tokens(
                    self.texts), self.wv_model.wv.key_to_index)

            # Get predictions and confidence scores.
            conf, preds = self.get_prediction(
                np.array(embeddings))

            # Generate output data
            data = [[self.labels[i], conf[i], preds[i]]
                    for i in range(len(self.labels))]

            # Clear the lists for the next batch.
            self.labels = []
            self.texts = []

            return data
        return 'BATCHING'
