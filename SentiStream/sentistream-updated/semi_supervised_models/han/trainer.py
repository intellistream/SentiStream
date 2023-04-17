# pylint: disable=import-error
# pylint: disable=no-name-in-module

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config

from semi_supervised_models.han.dataset import SentimentDataset
from semi_supervised_models.han.model import HAN
from semi_supervised_models.han.utils import calc_acc, join_tokens, preprocess
from utils import load_torch_model, downsampling


class Trainer:
    """
    Trainer class  to train Hierarchical Attention Network.
    """

    def __init__(self, docs, labels, wb_dict, embeddings, init, test_size=0.2, batch_size=128,
                 learning_rate=1e-3, word_hidden_size=50, sent_hidden_size=50, num_class=2,
                 early_stopping_patience=5, downsample=False):
        """
        Initialize class to train HAN.

        Args:
            docs (list): Documents of sentences.
            labels (list): Labels for each document.
            wb_dict (dict): Word to index dictionary.
            embeddings (list): List of word embeddings.
            init (bool): Flag indicating whether to initialize a new model or load from saved
                        weights.
            test_size (float): Fraction of the data to be used for validation. Defaults to 0.2.
            batch_size (int): Batch size for training. Defaults to 128.
            learning_rate (float): Learning rate for training. Defaults to 1e-3.
            word_hidden_size (int): Hidden state size for word-level attention layer.Defaults to 50.
            sent_hidden_size (int): Hidden state size for sentence-level attention layer. Defaults
                                    to 50.
            num_class (int): Number of classes for classification task. Defaults to 2.
            early_stopping_patience (int): Number of epochs to wait before early stopping. Defaults
                                             to 5.
            downsample (bool): Flag indicating whether to downsample the data to balance classes.
                                Defaults to False.
        """
        # Determine if GPU available for training.
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Optionally perform downsample to balance classes.
        if downsample:
            labels, docs = downsampling(labels, docs)

        max_word_length, max_sent_length = 15, 10
        self.early_stopping_patience = early_stopping_patience

        # Join all tokens into sentences to encode.
        docs = join_tokens(docs)

        embeddings = np.asarray(embeddings)

        # Get max sentence and word length for dataset.
        # if init:
        #     max_word_length, max_sent_length = get_max_lengths(
        #         docs)  # change to train only

        # Encode documents to model input format.
        docs = preprocess(docs, wb_dict,
                          max_word_length, max_sent_length)

        # Split data into training and validation sets.
        x_train, x_test, y_train, y_test = train_test_split(
            docs, labels, test_size=test_size, random_state=42)

        # Create PyTorch DataLoader objects for training and validation data.
        train_data, test_data = SentimentDataset(
            x_train, y_train), SentimentDataset(x_test, y_test)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0)  # due to effect of numpy, 0 give much faster loading.
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=0)

        # Initialize model and optimizer.
        if init:
            self.model = HAN(word_hidden_size, sent_hidden_size, batch_size, num_class,
                             embeddings, max_sent_length, max_word_length)
        else:
            self.model = load_torch_model(config.SSL_CLF)
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters(
        )), lr=learning_rate)

        # Initialize best model to None (will be updated during training).
        self.best_model = None

    def fit(self, epochs):
        """
        Train HAN for specified number of epochs and stores best model based on validation loss.

        Args:
            epochs (int): Number of epochs to train model.
        """
        # Initialize variables to keep track of best epoch and loss.
        best_epoch = 0
        best_loss = 1e5

        # Loop through number of epochs.
        for epoch in range(epochs):

            # Set model to training mode and initialize training loss and accuracy.
            self.model.train()
            train_loss = 0
            train_acc = 0

            # Loop through training data.
            for vecs, labels in self.train_loader:
                vecs, labels = vecs.to(self.device), labels.to(self.device)

                # Compute model output and loss, and update model parameters.
                self.optimizer.zero_grad()
                self.model.reset_hidden_state()
                pred = self.model(vecs)
                loss = self.criterion(pred, labels)
                loss.backward()
                self.optimizer.step()

                # Update training loss and accuracy.
                train_loss += loss.item()
                train_acc += calc_acc(labels, pred).item()

            # Compute average training loss and accuracy.
            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)

            # Set model to evaluation mode and initialize validation loss and accuracy.
            self.model.eval()
            val_loss = 0
            val_acc = 0

            with torch.no_grad():
                # Loop through the validation data.
                for vecs, labels in self.test_loader:
                    num_sample = len(labels)
                    vecs, labels = vecs.to(self.device), labels.to(self.device)

                    # Compute model output and loss.
                    self.model.reset_hidden_state(num_sample)
                    pred = self.model(vecs)

                    # Update validation loss and accuracy.
                    val_loss += self.criterion(pred, labels).item()
                    val_acc += calc_acc(labels, pred).item()

            # Compute average validation loss and accuracy.
            val_loss /= len(self.test_loader)
            val_acc /= len(self.test_loader)

            # Check if current model has the best validation loss so far and if it is then
            # update values.
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_epoch_details = f"epoch: {epoch+1}, train loss: {train_loss:.4f}, " \
                    f"train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                self.best_model = self.model

            # Check if the current epoch is more than 5 epochs away from the best epoch, if it is,
            # then stop training.
            if epoch - best_epoch > self.early_stopping_patience:
                print(best_epoch_details)
                break

    def fit_and_save(self, filename, epochs=100):
        """
        Train model and save best model.

        Args:
            filename (str): Filename to save model.
            epochs (int, optional): Number of epochs to train model. Defaults to 5=100.
        """
        # Train model.
        self.fit(epochs=epochs)

        # Set best model to eval mode.
        self.best_model.eval()

        # Save best model.
        torch.save(self.best_model, filename)
