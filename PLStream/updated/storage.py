from functools import lru_cache
import redis
import pickle
import numpy as np
from numpy import float32 as REAL
from collections import defaultdict


class VocabStorage():
    def __init__(self, ip='localhost', port=6379) -> None:
        pool = redis.ConnectionPool(host=ip, port=port, db=0)
        self.conn = redis.StrictRedis(connection_pool=pool)

        # redis key
        self.KEY_VOCAB = 'plstream_vocab'
        self.LRU_CHACHE_SIZE = 30000

        self.vocab = defaultdict(int)
        self.index_to_key = []
        self.key_to_index = {}

    def __len__(self):
        return len(self.index_to_key)

    def update_global_vocab(self, word, count):
        self.conn.hset(self.KEY_VOCAB, word, count)

    def merge_vocab(self):
        global_vocab = self.conn.hgetall(self.KEY_VOCAB)
        for k, v in global_vocab.items():
            key = str(k, encoding='utf-8')
            self.vocab[key] = max(self.vocab[key], int(v))

    def set_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence:
                if word not in self.vocab:
                    self.key_to_index[word] = len(self.index_to_key)
                    self.index_to_key.append(word)
                self.vocab[word] += 1

    def get_vocab(self):
        if np.random.random() < 0.001:
            self.merge_vocab()
        return self.vocab

    def get_index(self):
        return self.index_to_key, self.key_to_index


class ModelStorage():
    def __init__(self, vector_size, layer1_size, seed=1) -> None:
        """Save the necessary model parameters for sharing across operators.
        """
        self.vector_size = vector_size
        self.layer1_size = layer1_size
        self.rng = np.random.default_rng(seed=seed)
        self.vectors = np.zeros((0, vector_size), dtype=REAL)
        self.weights = np.zeros((0, vector_size), dtype=REAL)

        # constants
        self.ZERO_VEC = pickle.dumps(np.zeros(vector_size, dtype=REAL))

    def _make_vectors(self, tlen):
        vector = self.rng.random((tlen, self.vector_size), dtype=REAL)
        vector *= 2
        vector -= 1
        return vector

    def get_vectors(self, tlen):
        diff = tlen - len(self.vectors)
        if diff > 0:
            self.vectors = np.vstack([self.vectors, self._make_vectors(diff)])
        return self.vectors

    def set_vectors(self, vectors):
        self.vectors = vectors

    def get_weights(self, tlen):
        diff = tlen - len(self.weights)
        if diff > 0:
            self.weights = np.vstack(
                [self.weights, np.zeros((diff, self.layer1_size), dtype=REAL)])
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save_vectors(self):
        pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
        self.conn = redis.StrictRedis(connection_pool=pool)
        self.conn.set("plstream_vectors", pickle.dumps(self.vectors))
