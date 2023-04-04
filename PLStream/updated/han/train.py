# TODO: TEXT PREPROCESSING, STOP WORDS
import time
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config

from utils import get_max_lengths, clean_text, preprocess, calc_acc
from dataset import SentimentDataset
from hierarchical_att_model import HAN


def train():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    df = pd.read_csv('train.csv', names=['label', 'document'])
    df.label -= 1

    df = df[:1000]

    df['document'] = df['document'].apply(clean_text)

    wb_dict = pd.read_csv('glove.6B.50d.txt', header=None, sep=" ", quoting=3,
                          usecols=[0]).values.ravel()

    wb_dict = {val: idx for idx, val in enumerate(wb_dict)}

    max_word_length, max_sent_length = get_max_lengths(
        df.document)  # change to train only

    docs = preprocess(df['document'].tolist(), wb_dict,
                      max_word_length, max_sent_length)

    x_train, x_test, y_train, y_test = train_test_split(
        docs, df.label.tolist(), test_size=0.2, random_state=42)

    training_set = SentimentDataset(y_train, x_train)

    training_generator = DataLoader(
        training_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_set = SentimentDataset(y_test, x_test)
    test_generator = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    model = HAN(config.WORD_HIDDEN_SIZE, config.SENT_HIDDEN_SIZE, config.BATCH_SIZE, config.N_CLASS,
                'glove.6B.50d.txt', max_sent_length, max_word_length).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=config.LR)  # , momentum=config.MOMENTUM)
    best_loss = 1e5
    best_epoch = 0
    best_model = None

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0

        for vec, label in training_generator:
            vec = vec.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            pred = model(vec)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc += calc_acc(label, pred)
        train_loss /= len(training_generator)
        train_acc /= len(training_generator)

        model.eval()
        val_loss = 0
        val_acc = 0
        for vec, label in test_generator:
            num_sample = len(label)
            vec = vec.cuda()
            label = label.cuda()
            with torch.no_grad():
                model._init_hidden_state(num_sample)
                pred = model(vec)
            val_loss += criterion(pred, label)
            val_acc += calc_acc(label, pred)

        val_loss /= len(test_generator)
        val_loss = val_loss.item()
        val_acc /= len(test_generator)

        print(f"time: {time.time() - start_time} epoch: {epoch+1}, training loss: {train_loss:.4f}, training acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            print(f"Best loss {val_loss}")
            best_model = model

        if epoch - best_epoch > config.EARLY_STOPPING_PATIENCE:
            print(
                f"Stop training at epoch {epoch+1}. The lowest loss achieved is {val_loss}")
            print(best_epoch)
            break

    torch.save(best_model, "best_model.pth")


if __name__ == "__main__":
    train()
