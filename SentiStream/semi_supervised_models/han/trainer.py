# pylint: disable=import-error
# pylint: disable=no-name-in-module
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config

from semi_supervised_models.dataset import SentimentDataset
from semi_supervised_models.utils import (
    calc_acc, join_tokens, preprocess, get_max_lengths, downsampling)
from semi_supervised_models.han.model import HAN
from utils import load_torch_model


class Trainer:
    """
    Trainer class  to train Hierarchical Attention Network.
    """

    def __init__(self, docs, labels, wb_dict, embeddings, init, old_embeddings=None, test_size=0.2,
                 batch_size=512, learning_rate=0.002, word_hidden_size=32, sent_hidden_size=32,
                 early_stopping_patience=5):
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
            batch_size (int): Batch size for training. Defaults to 512.
            learning_rate (float): Learning rate for training. Defaults to 1e-3.
            word_hidden_size (int): Hidden state size for word-level attention layer.Defaults to 32.
            sent_hidden_size (int): Hidden state size for sentence-level attention layer. Defaults
                                    to 32.
            early_stopping_patience (int): Number of epochs to wait before early stopping. Defaults
                                             to 5.
        """
        # Determine if GPU available for training.
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')

        self.early_stopping_patience = early_stopping_patience

        # Downsample to balance classes.
        labels, docs = downsampling(labels, docs)

        # Join all tokens into sentences to encode.
        docs = join_tokens(docs)

        embeddings = np.asarray(embeddings)

        labels = torch.tensor(labels, dtype=torch.float32,
                              device=self.device).unsqueeze(1)

        max_word_length, max_sent_length = 17, 28
        # Get max sentence and word length for dataset.
        # if init:
        #     max_word_length, max_sent_length = get_max_lengths(
        #         docs)  # change to train only)
        #     print(max_word_length, max_sent_length)

        # Encode documents to model input format.
        docs = torch.from_numpy(
            np.array(preprocess(docs, wb_dict,
                                max_word_length, max_sent_length))).to(self.device)

        # Split data into training and validation sets.
        x_train, x_test, y_train, y_test = train_test_split(
            docs, labels, test_size=test_size, random_state=42)

        # Create PyTorch DataLoader objects for training and validation data.
        train_data, test_data = SentimentDataset(
            x_train, y_train), SentimentDataset(x_test, y_test)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0)
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=0)

        # Initialize model and optimizer.
        if init:
            self.model = HAN(embeddings, batch_size=batch_size,
                             max_sent_length=max_sent_length, max_word_length=max_word_length,
                             word_hidden_size=word_hidden_size, sent_hidden_size=sent_hidden_size)
        else:
            self.model, opt, scheduler = load_torch_model(
                HAN(np.array(old_embeddings)), config.SSL_CLF, train=True)

            embeddings = torch.from_numpy(np.concatenate(
                [np.zeros((1, embeddings.shape[1])), embeddings], axis=0).astype(np.float32))
            self.model.word_attention_net.lookup = torch.nn.Embedding.from_pretrained(
                embeddings)

        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters(
        )), lr=learning_rate)

        if not init:
            self.optimizer.load_state_dict(opt)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9)

        if not init:
            self.scheduler.load_state_dict(scheduler)

        self.best_model_checkpoint = None

    def fit(self, epochs):
        """
        Train HAN for specified number of epochs and stores best model based on validation loss.

        Args:
            epochs (int): Number of epochs to train model.
        """
        # Initialize variables to keep track of best epoch and loss.
        best_epoch = 0
        best_loss = 1e5

        # train_loss = [0] * epochs
        # train_acc = [0] * epochs
        val_loss = [0] * epochs
        # val_acc = [0] * epochs

        # Loop through number of epochs.
        for epoch in range(epochs):

            # Set model to training mode and initialize training loss and accuracy.
            self.model.train()

            # Loop through training data.
            for vecs, labels in self.train_loader:
                # Compute model output and loss, and update model parameters.
                self.optimizer.zero_grad()
                self.model.reset_hidden_state()
                pred = self.model(vecs)
                loss = self.criterion(pred, labels)
                loss.backward()
                self.optimizer.step()

            #     # Update training loss and accuracy.
            #     train_loss[epoch] += loss.item()
            #     train_acc[epoch] += calc_acc(pred, labels).item()

            # # Compute average training loss and accuracy.
            # train_loss[epoch] /= len(self.train_loader)
            # train_acc[epoch] /= len(self.train_loader)

            self.scheduler.step()

            # Set model to evaluation mode and initialize validation loss and accuracy.
            self.model.eval()

            with torch.no_grad():
                # Loop through the validation data.
                for vecs, labels in self.test_loader:
                    # Compute model output and loss.
                    self.model.reset_hidden_state(len(labels))
                    pred = self.model(vecs)

                    # Update validation loss and accuracy.
                    val_loss[epoch] += loss.item()
                    # val_acc[epoch] += calc_acc(pred, labels).item()

            # Compute average validation loss and accuracy.
            val_loss[epoch] /= len(self.test_loader)
            # val_acc[epoch] /= len(self.test_loader)

            # print(f"epoch: {epoch+1}, train loss: {train_loss[epoch]:.4f}, "
            #       f"train acc: {train_acc[epoch]:.4f}, val loss: {val_loss[epoch]:.4f},"
            #       f" val_acc: {val_acc[epoch]:.4f}, lr: {self.optimizer.param_groups[0]['lr']}")

            # Check if current model has the best validation loss so far and if it is then
            # update values.
            if val_loss[epoch] < best_loss:
                best_loss = val_loss[epoch]
                best_epoch = epoch
                # best_epoch_details = f"HAN epoch: {epoch+1},"\
                #     f" val loss: {val_loss[epoch]:.4f}"
                self.best_model_checkpoint = {'model_state_dict': self.model.state_dict(),
                                              'optimizer_state_dict': self.optimizer.state_dict(),
                                              'scheduler_state_dict': self.scheduler.state_dict()}

            # Check if the current epoch is more than 5 epochs away from the best epoch, if it is,
            # then stop training.
            if epoch - best_epoch > self.early_stopping_patience:
                # print(best_epoch_details)
                break

        # # Plot training and validation losses.
        # plt.subplot(2, 1, 1)
        # plt.plot(train_loss[:epoch+1], label='train')
        # plt.plot(val_loss[:epoch+1], label='val')
        # plt.title('Loss')
        # plt.legend()

        # # Plot training and validation accuracies.
        # plt.subplot(2, 1, 2)
        # plt.plot(train_acc[:epoch+1], label='train')
        # plt.plot(val_acc[:epoch+1], label='val')
        # plt.title('Accuracy')
        # plt.legend()

        # # Save the fig.
        # # plt.savefig(
        # #     f'{val_loss[best_epoch]}-{best_epoch}-{train_loss[best_epoch]}.png')
        # plt.savefig(
        #     'test.png')

    def fit_and_save(self, filename, epochs=50):
        """
        Train model and save best model.

        Args:
            filename (str): Filename to save model.
            epochs (int, optional): Number of epochs to train model. Defaults to 50.
        """
        # Train model.
        self.fit(epochs=epochs)

        # Save best model.
        torch.save(self.best_model_checkpoint, filename)
