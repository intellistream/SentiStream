from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import os
import random
import re
import numpy as np
from gensim.models import Word2Vec, FastText
from nltk.tokenize import sent_tokenize, word_tokenize
import torch

from nltk.stem import SnowballStemmer

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

stemmer = SnowballStemmer('english')


def process_batch(sentences):
    result = []

    for sent in sentences:
        result.append(clean(tokenize(sent)))

    return result


def clean(line):
    # return remove_stopwords(line)
    return [word for word in line if word not in STOP_WORDS]


def load_pseudo_data(folder_name):
    """Load ground truth and pseudo data to memory

    Parameters:
        folder_name (str): name of psedo data folder
        data_file (str): name of train/test data

    Returns:
        (tuple): tupe of length of pseudo data and combined dataframe to test
    """

    path_list = []
    for subdir_name in os.scandir(folder_name):
        for file_name in os.scandir(subdir_name):
            if file_name.is_file():
                path_list.append(file_name.path)

    pseudo_df = pd.concat(map(lambda path: pd.read_csv(
        path, delimiter='\t', header=None), path_list), ignore_index=True)
    pseudo_df.columns = ['label', 'review']

    return pseudo_df


def get_average_word_embeddings(model, tokens):
    avg_vec = np.zeros(model.vector_size)
    count = 0

    for word in tokens:
        if word in model.wv:
            avg_vec += model.wv[word]
            count += 1

    if count > 0:
        avg_vec /= count

    return avg_vec


def load_word_vector_model(algo, path):
    return algo.load(path)


def load_torch_model(path):
    model = torch.load(path)
    model.cuda()
    model.eval()

    return model


def downsampling(label, text):
    pos_idx = [idx for idx, x in enumerate(label) if x == 1]
    neg_idx = [idx for idx, x in enumerate(label) if x == 0]

    if len(pos_idx) < len(neg_idx):
        # no need to shuflle majority since already shuffled in train_test_split
        downsampled_idx = pos_idx + neg_idx[:len(pos_idx)]
    else:
        downsampled_idx = neg_idx + pos_idx[:len(neg_idx)]

    random.shuffle(downsampled_idx)

    return [label[i] for i in downsampled_idx], [text[i] for i in downsampled_idx]


def train_word_vector_algo(model, texts, path, update=True):
    model.build_vocab(texts, update=update)
    model.train(texts,
                total_examples=model.corpus_count,
                epochs=30 if not update else model.epochs)
    model.save(path)


def tokenize(text):
    text = re.sub(r"http\S+|www\S+|\@\w+", '', text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    # ?, !, . will be sentence stoppers
    text = re.sub(r'[^a-z.?!]+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = word_tokenize(text)

    return [stemmer(token) for token in tokens if token not in STOP_WORDS]


def clean_for_wv(tokens):
    return [token for token in tokens if len(token) > 1]
