import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(
            metrics.confusion_matrix(y_true, y_pred))
    return output


def mat_mul(input, weight, bias=None):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if bias is not None:
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def ele_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(docs):  # NO PREPROCESSING DONE
    words_length = []
    sents_length = []

    for doc in docs:
        sents = sent_tokenize(doc)
        sents_length.append(len(sents))

        for sent in sents:
            words_length.append(len(word_tokenize(sent)))

    return sorted(words_length)[int(0.8 * len(words_length))], sorted(sents_length)[int(0.8 * len(sents_length))]
