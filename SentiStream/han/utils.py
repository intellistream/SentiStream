import torch
import re
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
             "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
             'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
             'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o',
             're', 've', 'y', 'ma', 'st', 'nd', 'rd', 'th', "you'll", 'dr', 'mr', 'mrs']


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

    tokens = word_tokenize(text)

    # tokens = [stemmer.stem(token) for token in tokens]

    return [token for token in tokens if token not in STOP_WORDS]

def join_tokens(tokens):
    return (' '.join(tokens)).strip()


def calc_acc(y_label, y_pred):
    _, preds = torch.max(y_pred, 1)
    correct = torch.sum(preds == y_label)
    return correct.float() / preds.size(0)


def preprocess(docs, word_dict, max_length_word=15, max_length_sentences=15):
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


def get_max_lengths(docs):

    words_length = []
    sents_length = []

    for doc in docs:
        sents = sent_tokenize(doc)
        sents_length.append(len(sents))

        for sent in sents:
            words_length.append(len(word_tokenize(sent)))

    return sorted(words_length)[int(0.8 * len(words_length))], \
        sorted(sents_length)[int(0.8 * len(sents_length))]
