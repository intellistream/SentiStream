import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils import load_torch_model, downsampling

from han.utils import get_max_lengths, preprocess, clean_text_batch
from han import config
from han.hierarchical_att_model import HAN

class SentimentDataset(Dataset):
    def __init__(self, labels, texts):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

class Model:
    def __init__(self, doc, y, wb_dict, embeddings, init, test_size=0.2, batch_size=128):
        # y, x = downsampling(y, x)

        max_word_length = 15
        max_sent_length = 10

        doc = clean_text_batch(doc)

        # if init:
        #     max_word_length, max_sent_length = get_max_lengths(
        #         doc)  # change to train only

        docs = preprocess(doc, wb_dict,
                        max_word_length, max_sent_length)

        x_train, x_test, y_train, y_test = train_test_split(
            docs, y, test_size=0.2, random_state=42)

        training_set = SentimentDataset(y_train, x_train)

        self.training_generator = DataLoader(
            training_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        test_set = SentimentDataset(y_test, x_test)
        self.test_generator = DataLoader(
            test_set, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

        if init:
            self.model = HAN(config.WORD_HIDDEN_SIZE, config.SENT_HIDDEN_SIZE, config.BATCH_SIZE, config.N_CLASS,
                        embeddings, max_sent_length, max_word_length).cuda()
        else:
            self.model = load_torch_model('ssl-clf.pth').cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters(
        )), lr=config.LR)  # , momentum=config.MOMENTUM)
        
        self.best_model = None

    def fit(self, epoch):
        best_epoch = 0
        best_loss = 1e5

        for epoch in range(config.EPOCHS):
            # start_time = time.time()
            self.model.train()
            train_loss = 0
            train_acc = 0

            for vec, label in self.training_generator:
                vec = vec.cuda()
                label = label.cuda()
                self.optimizer.zero_grad()
                self.model._init_hidden_state()
                pred = self.model(vec)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

                # train_loss += loss
                # train_acc += calc_acc(label, pred)
            train_loss /= len(self.training_generator)
            train_acc /= len(self.training_generator)

            self.model.eval()
            val_loss = 0
            val_acc = 0
            for vec, label in self.test_generator:
                num_sample = len(label)
                vec = vec.cuda()
                label = label.cuda()
                with torch.no_grad():
                    self.model._init_hidden_state(num_sample)
                    pred = self.model(vec)
                val_loss += self.criterion(pred, label)
                # val_acc += calc_acc(label, pred)

            # val_loss /= len(test_generator)
            # val_loss = val_loss.item()
            # val_acc /= len(test_generator)

            # print(f"time: {time.time() - start_time} epoch: {epoch+1}, training loss: {train_loss:.4f}, training acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                # print(f"Best loss {val_loss}")
                self.best_model = self.model

            if epoch - best_epoch > config.EARLY_STOPPING_PATIENCE:
                # print(
                #     f"Stop training at epoch {epoch+1}. The lowest loss achieved is {best_loss}")
                # print(best_epoch)
                break

    def fit_and_save(self, filename, epoch=500):
        self.fit(epoch=epoch)
        self.best_model.eval()
        torch.save(self.best_model, filename)