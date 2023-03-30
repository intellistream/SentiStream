import pickle

from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
from os import walk
import os
import numpy as np
from gensim.models import Word2Vec
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


def clean(line):
    # return remove_stopwords(line)
    return [word for word in line if word not in STOP_WORDS]


def load_data(pseudo_data_folder, data_file):
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

    df = pd.read_csv(data_file, names=['label', 'review'])
    df['label'] -= 1

    return len(pseudo_df), pd.concat([df, pseudo_df], ignore_index=True)


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
    return torch.load(model_path, map_location=torch.device('cpu'))


def train_word2vec(model, sentences, path):
    # TODO: CHECK ON UPDATING MODEL VS RETRAINING FOR MIN_COUNT PRB
    model.build_vocab(sentences, update=True)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(path)
