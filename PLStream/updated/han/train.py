import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import shutil
import numpy as np

import config
from src.utils import get_max_lengths, get_evaluation
from src.dataset import SentimentDataset
from src.hierarchical_att_model import HAN

from sklearn.model_selection import train_test_split


def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    output_file = open("trained_models" + os.sep + "logs.txt", "w")

    df = pd.read_csv('train.csv', names=['label', 'document'])
    df.label -= 1

    df = df[:100]

    wb_dict = pd.read_csv('glove.6B.50d.txt', header=None, sep=" ", quoting=3,
                          usecols=[0]).values.ravel()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    max_word_length, max_sent_length = get_max_lengths(
        df.document)  # change to train only

    training_set = SentimentDataset(
        train_df.label, train_df.document, wb_dict, max_sent_length, max_word_length)
    training_generator = DataLoader(
        training_set, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    test_set = SentimentDataset(
        test_df.label, test_df.document, wb_dict, max_sent_length, max_word_length)
    test_generator = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=False)

    model = HAN(config.WORD_HIDDEN_SIZE, config.SENT_HIDDEN_SIZE, config.BATCH_SIZE, config.N_CLASS,
                'glove.6B.50d.txt', max_sent_length, max_word_length)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=config.LR, momentum=config.MOMENTUM)
    best_loss = 1e5
    best_epoch = 0
    num_iter_per_epoch = len(training_generator)
    for epoch in range(config.EPOCHS):
        model.train()

        start = time.time()
        for iter, (feature, label) in enumerate(training_generator):
            print(time.time() - start)
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(
            ), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                config.EPOCHS,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))

        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for te_feature, te_label in test_generator:
            num_sample = len(te_label)
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.no_grad():
                model._init_hidden_state(num_sample)
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=[
                                      "accuracy", "confusion_matrix"])
        output_file.write(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, config.EPOCHS,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            config.EPOCHS,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        if te_loss + config.EARLY_STOPPING_MIN_DELTA < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "trained_models" + os.sep + "whole_model_han")

        # Early stopping
        if epoch - best_epoch > config.EARLY_STOPPING_PATIENCE > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(
                epoch, te_loss))
            break


if __name__ == "__main__":
    train()
