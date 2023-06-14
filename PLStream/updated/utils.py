import heapq
import numpy as np

from numpy import dot
from numpy.linalg import norm
from nltk.stem import SnowballStemmer

from gensim.utils import simple_preprocess
from gensim.models.word2vec import Heapitem
from gensim.parsing.preprocessing import remove_stopwords


def make_input_collection(tag, X, y):
    tags = [tag] * len(X)
    data_ids = list(range(len(X)))
    collection = list(zip(tags, data_ids, X, y))

    return collection


def make_subsample(index_to_key, key_to_index, vocab, sample):
    vocab_size = len(index_to_key)
    sample_ints = np.zeros(vocab_size, dtype=np.uint32)
    try:
        total = sum([vocab[i] for i in index_to_key])
    except:
        return None

    if not sample:
        threshold_count = total
    elif sample < 1.0:
        threshold_count = sample * total
    else:
        threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

    for w in index_to_key:
        v = vocab[w]
        word_probability = (np.sqrt(v / threshold_count) +
                            1) * (threshold_count / v)
        sample_ints[key_to_index[w]] = np.uint32(
            word_probability * (2**32 - 1))

    return sample_ints


def make_cum_table(index_to_key, vocab, domain, ns_exponent):
    vocab_size = len(index_to_key)
    cum_table = np.zeros(vocab_size, dtype=np.uint32)

    train_words_pow = 0.0
    for word in index_to_key:
        count = vocab[word]
        train_words_pow += count**float(ns_exponent)

    cumulative = 0.0
    for idx, word in enumerate(index_to_key):
        count = vocab[word]
        cumulative += count**float(ns_exponent)
        cum_table[idx] = round(cumulative / train_words_pow * domain)

    if len(cum_table) > 0:
        assert cum_table[-1] == domain

    return cum_table


def build_heap(index_to_key, vocab):
    heap = list(Heapitem(vocab[word], i, None, None)
                for i, word in enumerate(index_to_key))
    heapq.heapify(heap)
    for i in range(len(index_to_key) - 1):
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap, Heapitem(count=min1.count + min2.count,
                           index=i + len(index_to_key), left=min1, right=min2))

    return heap


def make_huffman_tree(index_to_key, key_to_index, vocab):
    vocab_size = len(index_to_key)
    codes = np.zeros(vocab_size, dtype=object)
    points = np.zeros(vocab_size, dtype=object)

    heap = build_heap(index_to_key, vocab)

    max_depth = 0
    stack = [(heap[0], [], [])]
    while stack:
        node, codes, points = stack.pop()
        if node[1] < len(index_to_key):
            k = node[1]
            codes[key_to_index[k]] = codes
            points[key_to_index[k]] = points
            max_depth = max(len(codes), max_depth)
        else:
            points = np.array(
                list(points) + [node.index - len(index_to_key)], dtype=np.uint32)
            stack.append((node.left, np.array(
                list(codes) + [0], dtype=np.uint8), points))
            stack.append((node.right, np.array(
                list(codes) + [1], dtype=np.uint8), points))

    return codes, points


def compute_alpha(mina, maxa, progress):
    next_alpha = maxa - (maxa - mina) * progress
    next_alpha = max(mina, next_alpha)
    return next_alpha


def cos_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def clean_sentence(sentence, stemmer: SnowballStemmer):
    if stemmer is None:
        stemmer = SnowballStemmer('english')

    words = simple_preprocess(remove_stopwords(sentence), deacc=True)
    word_list = []
    for word in words:
        stem = stemmer.stem(word)
        if stem != '':
            word_list.append(stem)
    return word_list
