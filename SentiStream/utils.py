import pickle

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import logging
from os import walk
import os
import numpy as np
from gensim.models import Word2Vec


def tokenise(line):
    return simple_preprocess(line, deacc=True)
    # remove punctuations and lowercase words also tokenise them


def split(ls):
    for e in ls:
        yield e


def process(line):
    clean_text = clean(line)
    # logging.warning(line)
    tokenized_text = tokenise(clean_text)
    return tokenized_text


def clean(line):
    return remove_stopwords(line)


def load_and_augment_data(pseudo_data_folder, ground_data_file):
    # get pseudo data files
    files = []
    # pseudo_data_folder = './senti_output'
    for (dirpath, dirnames, filenames) in walk(pseudo_data_folder):
        filenames = [os.path.join(dirpath, f) for f in filenames]
        files.extend(filenames)

    # load pseudo data
    pdf = pd.DataFrame({'label': [], 'review': []})
    for file in files:
        tdf = pd.read_csv(file, header=None)
        tdf.columns = ["label", "review"]
        pdf = pdf.append(tdf, ignore_index=True)

    # tdf = pd.read_csv('./train.csv', header=None)  # , encoding='ISO-8859-1'
    new_df = pd.read_csv(ground_data_file, header=None)  # , encoding='ISO-8859-1'
    new_df.columns = ["label", "review"]
    pseudo_size = len(pdf)
    new_df.loc[new_df['label'] == 1, 'label'] = 0
    new_df.loc[new_df['label'] == 2, 'label'] = 1
    new_df = new_df.append(pdf, ignore_index=True)

    # test_df = pd.read_csv(ground_test_data_file, header=None)  # , encoding='ISO-8859-1'
    # test_df.columns = ["label", "review"]
    # pseudo_size = len(pdf)
    # test_df.loc[test_df['label'] == 1, 'label'] = 0
    # test_df.loc[test_df['label'] == 2, 'label'] = 1
    # test_df = test_df.append(pdf, ignore_index=True)

    return pseudo_size, new_df


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
    word_vector = []
    for token in tokenized_text:
        try:
            word_vector.append(model.wv[token])
        except:
            pass
    if len(word_vector) == 0:
        return np.zeros(model.vector_size)
    else:
        return (np.mean(word_vector, axis=0)).tolist()


def generate_vector_mean(model, tokenized_text, func=default_vector_mean):
    """
    :param model: pretrained model
    :param tokenized_text: list e.g. "['eat','rice']"
    :param func: custom function to generate vector mean with
    :return: vector mean in the form of list e.g. [0.1,0.2,0.4]
    """
    return func(model, tokenized_text)


def default_model_pretrain():
    path_to_model = 'word2vec20tokenised.model'
    return Word2Vec.load(path_to_model)


def default_model_classifier():
    path_to_model = 'randomforest_classifier'
    file = open(path_to_model, 'rb')
    return pickle.load(file)


def train_word2vec(model, sentences):
    model.build_vocab(sentences, update=True)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
