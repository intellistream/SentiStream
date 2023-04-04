import torch
import re
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize


def mat_mul(input, weight, bias=None):
    output = torch.matmul(input, weight)
    if bias is not None:
        output += bias
    output = torch.tanh(output)
    return output.squeeze()


def ele_wise_mul(input1, input2):
    output = input1 * input2.unsqueeze(2)
    return output.sum(dim=0, keepdim=True)


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|\@\w+", '', text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r'[^\w\s.?!]', '', text)  # ?, !, . will be sentence stoppers
    text = re.sub('\.+', '.', text)
    text = re.sub('\s+', ' ', text)

    # tokens = [token for token in tokens if token not in STOP_WORDS]##### CHECK WITHOUT STOP WORDS
    # tokens = [stemmer.stem(token) for token in tokens]

    return text.strip()


def calc_acc(y_label, y_pred):
    _, preds = torch.max(y_pred, 1)
    correct = torch.sum(preds == y_label)
    return correct.float() / preds.size(0)


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
            words_length.append(len(word_tokenize(sent)))

    return sorted(words_length)[int(0.8 * len(words_length))], \
        sorted(sents_length)[int(0.8 * len(sents_length))]
