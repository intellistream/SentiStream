import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import load_torch_model, downsampling


class SentimentDataset(Dataset):
    def __init__(self, vector, label):
        self.vector = vector
        self.label = label

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, idx):
        return self.vector[idx], self.label[idx]


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class Model:
    def __init__(self, x, y, input_dim, init, test_size=0.2, batch_size=16):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        ####
        # print("RAW: ", y.count(0), y.count(1))

        y, x = downsampling(y, x)

        # print("DOWNSAMPLING: ", y.count(0), y.count(1))

        if self.device == 'cpu':
            x = torch.FloatTensor(np.array(x))
            y = torch.FloatTensor(np.array(y)).unsqueeze(1)
        else:
            x = torch.cuda.FloatTensor(np.array(x))
            y = torch.cuda.FloatTensor(np.array(y)).unsqueeze(1)

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=test_size, random_state=42)

        train_data = SentimentDataset(x_train, y_train)
        val_data = SentimentDataset(x_val, y_val)

        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        self.val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False)

        if init:
            self.torch_model = Classifier(input_dim=input_dim, hidden_dim=32).to(self.device)
        else:
            self.torch_model = load_torch_model('ssl-clf.pth')

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            params=self.torch_model.parameters(), lr=5e-3)

        self.best_model = None

    def binary_acc(self, y_pred, y_test):
        """Calculate accuracy

        Args:
            y_pred (Tensor): Predicted results
            y_test (Tensor): Ground truth

        Returns:
            Tensor: Sum of correct predictions
        """
        y_h = torch.round(y_pred)
        crct_results = (y_h == y_test).sum()
        return crct_results / y_pred.size(0)

    def fit(self, epoch):
        best_epoch = 0
        best_loss = 1e5
        best_acc = 0
        for epoch in range(epoch):
            self.torch_model.train()
            # train_loss = 0
            # train_acc = 0

            for vecs, labels in self.train_loader:

                vecs = vecs.to(self.device)
                labels = labels.to(self.device)


                outputs = self.torch_model(vecs)
                loss = self.criterion(outputs, labels)
                acc = self.binary_acc(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # train_loss += loss.float()
                # train_acc += acc.float()

            # train_loss /= len(self.train_loader)
            # train_acc /= len(self.train_loader)

            self.torch_model.eval()
            val_loss = 0
            val_acc = 0

            with torch.no_grad():
                for vecs, labels in self.val_loader:
                    vecs = vecs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.torch_model(vecs)
                    loss = self.criterion(outputs, labels)
                    acc = self.binary_acc(outputs, labels)

                    val_loss += loss.float()
                    val_acc += acc.float()

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)

            if best_loss - val_loss > 0.001:
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc
                self.best_model = self.torch_model

            if epoch - best_epoch > 5:
                print(f'epoch: {best_epoch+1}, val_acc: {best_acc:.2f}, val_loss: {best_loss:.2f}')
                break

    def fit_and_save(self, filename, epoch=500):
        self.fit(epoch=epoch)
        self.best_model.eval()
        torch.save(self.best_model, filename)