# pylint: disable=import-error

import random
import re
import numpy as np
import nltk
import torch
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

import config

nltk.download('punkt', quiet=True)


STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
              "you've", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
              'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
              'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
              'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
              'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
              'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
              'ma', 'st', 'nd', 'rd', 'th', "you'll", 'dr', 'mr', 'mrs']

stemmer = SnowballStemmer('english')


def get_average_word_embeddings(model, docs):
    """
    Calcualte average word embeddings for list of docs using word vector model.

    Args:
        model (class): Word vector model
        docs (array-like): List of docs of tokens.

    Returns:
        ndarray: Average word embeddings for the input documents.
    """
    filtered_tokens = [
        [token for token in doc if token in model.wv.key_to_index]
        for doc in docs]

    doc_embeddings = np.zeros((len(filtered_tokens), model.vector_size))

    for idx, tokens in enumerate(filtered_tokens):
        if tokens:
            doc_embeddings[idx] = np.mean(model.wv[tokens], axis=0)

    return doc_embeddings


def load_torch_model(model, path):
    """
    Load PyTorch model to GPU for inference.

    Args:
        model (nn.Module): Model class to load state dict.
        path (str): Model path.

    Returns:
        Torch.nn.Module: Loaded PyTorch model
    """
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def downsampling(label, text):
    """
    Downsample majority class in binary classification to balance class.

    Args:
        label (list): List of labels.
        text (list): List of documents.

    Returns:
        tuple: Downsampled labels and documents.
    """
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
    """
    Train word vector algorithm and save it locally.

    Args:
        model (class): Intialized instance of word vector model. (Either Word2Vec or FastText).
        texts (list): List of tokens from documents.
        path (str): Path to save trained model.
        update (bool, optional): Flag indicating whether to update pretrained model.
                                Defaults to True.
    """
    model.build_vocab(texts, update=update)
    model.train(texts,
                total_examples=model.corpus_count,
                epochs=30 if not update else model.epochs)
    model.save(path)


def tokenize(text):
    """
    Clean and tokenize text for processing.

    Args:
        text (str): Text/Review to be tokenized.

    Returns:
        list: List of cleaned tokens generated from text.
    """
    # Remove URLs, tags.
    text = re.sub(r"http\S+|www\S+|\@\w+", '', text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    # Replace anything other than alphabets -- ?, !, . will be sentence stoppers -- needed for
    # sentence tokenization.
    text = re.sub(r'[^a-z.?!]+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    return [stemmer.stem(token) if config.STEM else token
            for token in tokens if token not in STOP_WORDS]


def clean_for_wv(tokens):
    """
    Clean unneccesary/meaningless tokens from generated tokens.

    Args:
        tokens (list): List of tokens from documents.

    Returns:
        list: List of filtered tokens.
    """
    return [token for token in tokens if len(token) > 1]
