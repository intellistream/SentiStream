# pylint: disable=import-error
# pylint: disable=no-name-in-module
import csv
import torch

from torch.utils.data import DataLoader

import config

from other_exp.ann.model import ANN
from semi_supervised_models.dataset import SentimentDataset
from sklearn.model_selection import train_test_split
from other_exp.utils import tokenize
from utils import get_average_word_embeddings


class Trainer:
    def __init__(self, wv_model, test_size=0.2, batch_size=256, hidden_size=32, learning_rate=3e-3):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        texts = []
        labels = []

        with open(config.TRAIN_DATA, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                texts.append(tokenize(row[2]))
                labels.append(int(row[1]))

        vectors = get_average_word_embeddings(wv_model, texts)

        vectors = torch.tensor(
            vectors, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32,
                              device=self.device).unsqueeze(1)

        x_train, x_val, y_train, y_val = train_test_split(
            vectors, labels, test_size=test_size, random_state=42)

        train_data, test_data = SentimentDataset(
            x_train, y_train), SentimentDataset(x_val, y_val)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0)
        self.test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=0)

        self.model = ANN(wv_model.vector_size, hidden_size)

        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate)

        self.sheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.9)

        self.best_model = None

    def fit(self, epochs):
        best_epoch = 0
        best_loss = 1e5

        val_loss = [0] * epochs

        for epoch in range(epochs):
            self.model.train()

            for vecs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(vecs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.sheduler.step()
            self.model.eval()

            with torch.no_grad():
                for vecs, labels in self.test_loader:
                    outputs = self.model(vecs)
                    loss = self.criterion(outputs, labels)
                    val_loss[epoch] += loss.item()

            val_loss[epoch] /= len(self.test_loader)

            if best_loss - val_loss[epoch] > 0.001:
                best_epoch = epoch
                best_loss = val_loss[epoch]

            if epoch - best_epoch > 5:
                self.best_model = self.model
                break

    def fit_and_save(self, epochs=100):
        self.fit(epochs=epochs)
        return self.best_model
