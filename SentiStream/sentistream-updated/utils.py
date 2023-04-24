# pylint: disable=import-error

import random
import re
import numpy as np
import nltk
import torch
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

STOP_WORDS = {'also', 'ltd', 'once', 'll', 'make', 'he', 'through', 'all', 'top', 'from', 'or', 's',
              'hereby', 'so',  'yours', 'since', 'meanwhile', 're', 'over', 'mrs', 'thereafter',
              'ca', 'move', 'mill', 'such', 'wherever', 'on', 'besides', 'few', 'does', 'yet',  'y',
              'much', 'my', 'him', 'yourselves', 'as', 'ours', 'therefore', 'amongst', 'due', 'mr',
              'here', 'may', 'onto', 'it', 'whose', 'himself', 'least', 'i', 'what', 'many', 'd',
              'hereafter', 'anything', 'of', 'whoever', 'made', 'be', 'sometimes', 'put', 'found',
              'than', 'although', 'anyway', 'seems', 'you', 'under', 'above', 'themselves', 'thus',
              'a', 'con', 'when', 'why', 'back', 'until', 'first', 'theirs', 'describe', 'because',
              'always', 'too', 'across', 't', 'anyhow', 'her', 'ourselves', 'latterly', 'six', 'an',
              'somewhere', 'else', 'for', 'really', 'up', 'among', 'used', 'whenever', 'during',
              'nowhere', 'nothing', 'if', 'afterwards', 'that', 'whereas', 'elsewhere', 'along',
              'been', 'both', 'etc', 'ie', 'might', 'into', 'inc', 'with', 'formerly', 'there',
              'will', 'own', 'seemed', 'though', 'was', 'whereupon', 'just', 'except', 'has',
              'your', 'do', 'around', 'herein', 'anywhere', 'rd',  'now', 'sincere', 'this',  'me',
              'throughout',  'unless', 'against', 'out', 'most', 'various', 'others', 'them', 'th',
              'eleven', 'am', 'indeed', 'name', 'his', 'often', 'yourself', 'only', 'kg', 'take',
              'everything', 'cry', 'and',  'quite', 'itself', 'in', 'to', 'well', 'namely', 'thru',
              'see', 'would', 'which', 'beforehand', 'myself', 'having', 'however', 'go', 'did',
              'below', 'those', 'st', 'computer', 'several', 'whether', 'have', 'between', 'any',
              'becoming', 'thereby', 'while', 'were', 'whole', 'latter', 'but', 'km', 'amount',
              'either', 'herself', 'whereafter', 'never', 'system', 'un', 'find', 'please', 'o',
              'hereupon', 'thin', 'give', 'third', 'every', 'doing', 'our', 'towards', 'another',
              'before', 'within', 'mine', 'almost', 'mostly', 'down', 'de', 'seeming', 'moreover',
              'some', 'us', 'former', 'call', 'should', 'she', 'even', 'beyond', 'became', 'other',
              'show', 'eg', 'about', 'side', 'its', 'these', 'rather', 'alone', 'nd', 'after',
              'already', 'keep', 'more', 'behind', 'thick', 'together', 'upon', 'interest', 'dr',
              'otherwise', 'full', 'can', 'next', 'last', 'bill', 'their', 'hers', 'hence', 'by',
              'become', 'something', 'who', 'further', 'someone', 'must', 'say', 'each', 'very',
              'whom', 'again', 'then', 'we', 'same', 'via', 'where', 'per', 'are', 'the', 'still',
              'toward', 'anyone', 'therein', 'being', 'off', 'perhaps', 'is', 'had', 'co', 'at',
              'done', 'everywhere', 'less', 'wherein', 'could', 'ma', 'sometime', 'seem', 'somehow',
              'beside', 'whatever', 'whereby', 'ever', 'everyone', 'nevertheless', 'serious',
              'using', 'becomes', 'enough', 'how', 'bottom', 've', 'regarding', 'm', 'they', 'part',
              'front', 'fill', 'get', 'nobody', 'detail'}

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


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
    text = text.replace('\\n', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\r', ' ')
    # Replace anything other than alphabets -- ?, !, . will be sentence stoppers -- needed for
    # sentence tokenization.
    text = re.sub(r'[^a-z.?!]+', ' ', text)
    text = re.sub(r'\.+', '.', text)
    text = text.replace('.', ' . ')
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    # TODO: CHECK
    tokens = [lemmatizer.lemmatize(token)
              for token in tokens if token not in STOP_WORDS]

    for i, token in enumerate(tokens[:-1]):
        if token == 'not':
            tokens[i:i+2] = ['n_' + tokens[i+1]] + \
                [''] if i < len(tokens) - 1 else ['n_' + tokens[i+1]]

            # print(tokens[i])

    # return [stemmer.stem(token) if config.STEM else token
    #         for token in tokens if token not in STOP_WORDS]
    # return [lemmatizer.lemmatize(token) if config.STEM else token
    #         for token in tokens if token not in STOP_WORDS]
    return tokens


def clean_for_wv(tokens):
    """
    Clean unneccesary/meaningless tokens from generated tokens.

    Args:
        tokens (list): List of tokens from documents.

    Returns:
        list: List of filtered tokens.
    """
    return [token for token in tokens if len(token) > 1]
