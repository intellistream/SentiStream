# pylint: disable=import-error
import multiprocessing
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from semi_supervised_models.ann.dataset import SentimentDataset
from semi_supervised_models.ann.model import Classifier
from semi_supervised_models.ann.utils import calc_acc
from utils import load_torch_model, downsampling

class Trainer:
    """
    Trainer class  to train simple feed forward network.
    """
    def __init__(self, vectors, labels, input_size, init, test_size=0.2, batch_size=16,
                 hidden_size=32, learning_rate=5e-3, downsample=False):
        """
        Initialize class to train classifier

        Args:
            vectors (array-like): Input word vectors.
            labels (array-like): Corresponding labels.
            input_size (int): Size of the input layer of neural network.
            init (bool): Flag indicating whether to initialize new model or train an existing one.
            test_size (float, optional): Fraction of the data to be used for validation.
                                        Defaults to 0.2.
            batch_size (int, optional): Batch size for training. Defaults to 16.
            hidden_size (int, optional): Size of the hidden layer of neural network. Defaults to 32.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 5e-3.
            downsample (bool, optional): Flag indicating whether to downsample to balance classes.
                                        Defaults to False.
        """
        # Determine if GPU available for training.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Optionally perform downsample to balance classes.
        if downsample:
            labels, vectors = downsampling(labels, vectors)

        # Convert data to PyTorch tensors and move to device.
        vectors = torch.FloatTensor(np.array(vectors)).to(self.device)
        labels = torch.FloatTensor(np.array(labels)).unsqueeze(1).to(self.device)

        # Split data into training and validation sets.
        x_train, x_val, y_train, y_val = train_test_split(
            vectors, labels, test_size=test_size, random_state=42)

        # Create PyTorch DataLoader objects for training and validation data.
        num_workers = multiprocessing.cpu_count()
        train_data, test_data = SentimentDataset(x_train, y_train), SentimentDataset(x_val, y_val)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers)
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers)

        # Initialize model and optimizer.
        if init:
            self.model = Classifier(input_size, hidden_size)
        else:
            self.model = load_torch_model('ssl-clf.pth')
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

        # Initialize best model to None (will be updated during training).
        self.best_model = None

    def fit(self, epochs):
        """
        Train ANN for specified number of epochs and stores best model based on validation loss.

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
                outputs = self.model(vecs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Update training loss and accuracy.
                train_loss += loss.item()
                train_acc += calc_acc(outputs, labels).item()

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
                    vecs, labels = vecs.to(self.device), labels.to(self.device)

                    # Compute model output and loss.
                    outputs = self.model(vecs)
                    loss = self.criterion(outputs, labels)

                    # Update validation loss and accuracy.
                    val_loss += loss.item()
                    val_acc += calc_acc(outputs, labels).item()

            # Compute average validation loss and accuracy.
            val_loss /= len(self.test_loader)
            val_acc /= len(self.test_loader)

            # Check if current model has the best validation loss so far and if it is then
            # update values.
            if best_loss - val_loss > 0.001:
                best_epoch = epoch
                best_loss = val_loss
                best_epoch_details = f"epoch: {epoch+1}, train loss: {train_loss:.4f}, " \
                    "train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                self.best_model = self.model

            # Check if the current epoch is more than 5 epochs away from the best epoch, if it is,
            # then stop training.
            if epoch - best_epoch > 5:
                print(best_epoch_details)
                break

    def fit_and_save(self, filename, epochs=500):
        """
        Train model and save best model.

        Args:
            filename (str): Filename to save model.
            epochs (int, optional): Number of epochs to train model. Defaults to 500.
        """
        # Train model.
        self.fit(epochs=epochs)

        # Set best model to eval mode.
        self.best_model.eval()

        # Save best model.
        torch.save(self.best_model, filename)
