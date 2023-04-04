import torch
import re
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


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|\@\w+", '', text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r'[^\w\s.?!]', '', text)  # ?, !, . will be sentence stoppers
    text = re.sub('\.+', '.', text)
    text = re.sub('\s+', ' ', text)

    # tokens = [token for token in tokens if token not in STOP_WORDS] ##### CHECK WITHOUT STOP WORDS
    # tokens = [stemmer.stem(token) for token in tokens]

    return text.strip()


def preprocess(docs, word_dict, max_length_word, max_length_sentences):
    padded_words = [-1] * max_length_word

    temp = []

    for doc in docs:

        # UNK = -1 , PAD = -1  ### HAVE SEPARATE ENCODINGSSS

        document_encode = [
            [word_dict.get(word, -1) for word in word_tokenize(sentences)] for sentences
            in
            sent_tokenize(doc)]

        for sentence in document_encode:
            if len(sentence) < max_length_word:
                sentence += padded_words[len(sentence):]

        if len(document_encode) < max_length_sentences:
            document_encode += [padded_words] * \
                (max_length_sentences - len(document_encode))

        document_encode = [sentences[:max_length_word] for sentences in document_encode][
            :max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1  # make all pos

        temp.append(document_encode)

    return temp


def get_max_lengths(docs):  # NO PREPROCESSING DONE

    words_length = []
    sents_length = []

    for doc in docs:
        sents = sent_tokenize(doc)
        sents_length.append(len(sents))

        for sent in sents:
            # words_length.append(len(tokenizer.tokenize(sent)))
            words_length.append(len(word_tokenize(sent)))

    return sorted(words_length)[int(0.8 * len(words_length))], sorted(sents_length)[int(0.8 * len(sents_length))]
