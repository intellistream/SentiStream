from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import os
import random
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import torch

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


def tokenize(line):
    # TODO: Change Min, Max LEN ###############################################################################################################
    return simple_preprocess(line, deacc=True)
    # remove punctuations and lowercase words also tokenise them

def process(line):
    # clean_text = clean(line)
    # tokenized_text = tokenise(clean_text)

    tokenized_text = tokenize(line)
    tokens = clean(tokenized_text)

    return tokens

def process_batch(sentences):
    result = []

    for sent in sentences:
        result.append(clean(tokenize(sent)))
    
    return result


def clean(line):
    # return remove_stopwords(line)
    return [word for word in line if word not in STOP_WORDS]


def load_data(pseudo_data_folder):
    """Load ground truth and pseudo data to memory

    Parameters:
        pseudo_data_folder (str): name of psedo data folder
        data_file (str): name of train/test data

    Returns:
        (tuple): tupe of length of pseudo data and combined dataframe to test
    """

    path_list = []
    for subdir_name in os.scandir(pseudo_data_folder):
        for file_name in os.scandir(subdir_name):
            if file_name.is_file():
                path_list.append(file_name.path)

    pseudo_df = pd.concat(map(lambda path: pd.read_csv(
        path, delimiter='\t', header=None), path_list), ignore_index=True)
    pseudo_df.columns = ['label', 'review']
    
    return pseudo_df


def pre_process(tweet, func=process):
    """

    :param tweet:expects tweet in the format of a label,string: 1,"i love rice"
    :param func: funct(text) returns tokenized text in the form of a list. e.g: ['eat','rice']
    :return: label,[tokens]
    """
    return tweet[0], process_text_and_generate_tokens(tweet[1], func)


def process_text_and_generate_tokens(text, func=process):
    """
    :param func: funct(text) returns tokenized text in the form of a list. e.g: ['eat','rice']
    :param text: expects text in the format of a string:"i eat rice"
    :return: [tokens]
    """

    return func(text)


def default_vector_mean(model, tokenized_text):
    """Calculate average word embedding

    Parameters:
        model (T): word vector model
        tokenized_text (list): list of tokenized words

    Returns:
        ndarray: average word vector
    """
    word_vector = np.zeros(model.vector_size)
    count = 0
    for token in tokenized_text:
        try:
            word_vector += model.wv[token]
            count += 1
        except:
            pass

    if count > 0:
        word_vector /= count

    return word_vector


def generate_vector_mean(model, tokenized_text, func=default_vector_mean):
    """
    :param model: pretrained model
    :param tokenized_text: list e.g. "['eat','rice']"
    :param func: custom function to generate vector mean with
    :return: vector mean in the form of list e.g. [0.1,0.2,0.4]
    """
    return func(model, tokenized_text)


def default_model_pretrain(path_to_model='word2vec20tokenised.model'):
    return Word2Vec.load(path_to_model)

def load_torch_model(model_path):
    """Load trained torch model on to cpu (compatible with both models trained on cpu & gpu)

    Parameters:
        model_path (str): path of torch model

    Returns:
        any: load all tensors to cpu
    """
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    return model

def downsampling(label, text):
    pos_idx = [idx for idx, x in enumerate(label) if x == 1]
    neg_idx = [idx for idx, x in enumerate(label) if x == 0]

    if len(pos_idx) < len(neg_idx):
        downsampled_idx = pos_idx + neg_idx[:len(pos_idx)] # no need to shuflle majority since already shuffled in train_test_split
    else:
        downsampled_idx = neg_idx + pos_idx[:len(neg_idx)]

    random.shuffle(downsampled_idx)

    return [label[i] for i in downsampled_idx], [text[i] for i in downsampled_idx]

def train_word2vec(model, sentences, path):
    # TODO: CHECK ON UPDATING MODEL VS RETRAINING FOR MIN_COUNT PRB
    model.build_vocab(sentences, update=True)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(path)

def clean_text_w2v(text):
    text = re.sub(r"http\S+|www\S+|\@\w+", '', text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r'[^\w\s.?!]', '', text)  # ?, !, . will be sentence stoppers
    text = re.sub('\.+', '.', text)
    text = re.sub('\s+', ' ', text)

    tokens = word_tokenize(text)

        # tokens = [stemmer.stem(token) for token in tokens]

    return [token for token in tokens if token not in STOP_WORDS]


def clean_text_han(doc):
    temp = []

    for tokens in doc:
             temp.append((' '.join(tokens)).strip())

    return temp

def preprocess_han(docs, word_dict, max_length_word=15, max_length_sentences=10):
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

def mat_mul(input, weight, bias=None):
    output = torch.matmul(input, weight)
    if bias is not None:
        output += bias
    output = torch.tanh(output)
    return output.squeeze()


def ele_wise_mul(input1, input2):
    output = input1 * input2.unsqueeze(2)
    return output.sum(dim=0, keepdim=True)

def calc_acc(y_label, y_pred):
    _, preds = torch.max(y_pred, 1)
    correct = torch.sum(preds == y_label)
    return correct.float() / preds.size(0)
