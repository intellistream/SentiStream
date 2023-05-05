# pylint: disable=import-error
# pylint: disable=no-name-in-module
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config

from semi_supervised_models.dataset import SentimentDataset
from semi_supervised_models.utils import calc_acc, downsampling
from semi_supervised_models.ann.model import Classifier
from utils import load_torch_model


class Trainer:
    """
    Trainer class  to train simple feed forward network.
    """

    def __init__(self, vectors, labels, input_size, init, test_size=0.2, batch_size=256,
                 hidden_size=32, learning_rate=4e-3):
        """
        Initialize class to train classifier

        Args:
            vectors (array-like): Input word vectors.
            labels (array-like): Corresponding labels.
            input_size (int): Size of the input layer of neural network.
            init (bool): Flag indicating whether to initialize new model or train an existing one.
            test_size (float, optional): Fraction of the data to be used for validation.
                                        Defaults to 0.2.
            batch_size (int, optional): Batch size for training. Defaults to 256.
            hidden_size (int, optional): Size of the hidden layer of neural network. Defaults to 32.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 3e-3.
        """
        # Determine if GPU available for training.
        # self.device = torch.device(
        #     'cuda:0' if torch.cuda.is_available() else 'cpu')
        # CPU is faster than GPU for ANN -- might be simple model.
        # TODO: Remove to(device) when finialized with CPU
        self.device = 'cpu'

        # Downsample to balance classes.
        labels, vectors = downsampling(labels, vectors)

        vectors = np.array(vectors)

        # Convert data to PyTorch tensors move directly to device.
        vectors = torch.tensor(
            vectors, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32,
                              device=self.device).unsqueeze(1)

        # Split data into training and validation sets.
        x_train, x_val, y_train, y_val = train_test_split(
            vectors, labels, test_size=test_size, random_state=42)

        # Create PyTorch DataLoader objects for training and validation data.
        train_data, test_data = SentimentDataset(
            x_train, y_train), SentimentDataset(x_val, y_val)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0)
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=0)

        # Initialize model and optimizer.
        if init:
            self.model = Classifier(input_size, hidden_size)
        else:
            self.model, opt, scheduler = load_torch_model(Classifier(
                input_size, hidden_size), config.SSL_CLF, train=True)
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)

        if not init:
            self.optimizer.load_state_dict(opt)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9)

        if not init:
            self.scheduler.load_state_dict(scheduler)

        # Initialize best model to None (will be updated during training).
        self.best_model_checkpoint = None

    def fit(self, epochs):
        """
        Train ANN for specified number of epochs and stores best model based on validation loss.

        Args:
            epochs (int): Number of epochs to train model.
        """
        # Initialize variables to keep track of best epoch and loss.
        best_epoch = 0
        best_loss = 1e5

        train_loss = [0] * epochs
        train_acc = [0] * epochs
        val_loss = [0] * epochs
        val_acc = [0] * epochs

        # Loop through number of epochs.
        for epoch in range(epochs):

            # Set model to training mode and initialize training loss and accuracy.
            self.model.train()

            # Loop through training data.
            for vecs, labels in self.train_loader:
                # Compute model output and loss, and update model parameters.
                self.optimizer.zero_grad()
                outputs = self.model(vecs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Update training loss and accuracy.
                train_loss[epoch] += loss.item()
                train_acc[epoch] += calc_acc(outputs, labels).item()

            # Compute average training loss and accuracy.
            train_loss[epoch] /= len(self.train_loader)
            train_acc[epoch] /= len(self.train_loader)

            self.scheduler.step()

            # Set model to evaluation mode and initialize validation loss and accuracy.
            self.model.eval()

            with torch.no_grad():
                # Loop through the validation data.
                for vecs, labels in self.test_loader:
                    # Compute model output and loss.
                    outputs = self.model(vecs)
                    loss = self.criterion(outputs, labels)

                    # Update validation loss and accuracy.
                    val_loss[epoch] += loss.item()
                    val_acc[epoch] += calc_acc(outputs, labels).item()

            # Compute average validation loss and accuracy.
            val_loss[epoch] /= len(self.test_loader)
            val_acc[epoch] /= len(self.test_loader)

            # print(f"epoch: {epoch+1}, train loss: {train_loss[epoch]:.4f}, "
            #       f"train acc: {train_acc[epoch]:.4f}, val loss: {val_loss[epoch]:.4f}, "
            #       f"val_acc: {val_acc[epoch]:.4f}, lr: {self.optimizer.param_groups[0]['lr']}")

            # Check if current model has the best validation loss so far and if it is then
            # update values.
            if best_loss - val_loss[epoch] > 0.001:
                best_epoch = epoch
                best_loss = val_loss[epoch]
                best_epoch_details = f"ANN epoch: {epoch+1}," \
                    f" train loss: {train_loss[epoch]:.4f}, train acc: {train_acc[epoch]:.4f}," \
                    f" val loss: {val_loss[epoch]:.4f}, val_acc: {val_acc[epoch]:.4f}"
                self.best_model_checkpoint = {'model_state_dict': self.model.state_dict(),
                                              'optimizer_state_dict': self.optimizer.state_dict(),
                                              'scheduler_state_dict': self.scheduler.state_dict()}

            # Check if the current epoch is more than 5 epochs away from the best epoch, if it is,
            # then stop training.
            if epoch - best_epoch > 10:
                print(best_epoch_details)
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

    def fit_and_save(self, filename, epochs=100):
        """
        Train model and save best model.

        Args:
            filename (str): Filename to save model.
            epochs (int, optional): Number of epochs to train model. Defaults to 500.
        """
        # Train model.
        self.fit(epochs=epochs)

        # # Save best model.
        torch.save(self.best_model_checkpoint, filename)
