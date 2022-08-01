from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import logging
from os import walk
import os


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


def pre_process(tweet):
    return tweet[0], process(tweet[1])
