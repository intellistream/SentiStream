import time
import nltk
import logging
import numpy as np
from numpy import float32 as REAL
from pandas import read_csv
from json import loads, dumps

from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec_inner import (
    train_batch_sg,
    train_batch_cbow
)

from sklearn.metrics import accuracy_score, f1_score

from storage import ModelStorage, VocabStorage
from config import Word2VecConfig
from utils import (
    compute_alpha,
    make_subsample,
    make_huffman_tree,
    make_cum_table,
    cos_similarity,
    clean_sentence,
    make_input_collection
)

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.datastream.connectors import StreamingFileSink
from pyflink.common.typeinfo import Types
from pyflink.common.serialization import Encoder

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)

LABEL_NEG, LABEL_POS = 0, 1
BATCHING_FLAG = 'BATCHING'


class Word2VecModel:
    def __init__(self, config: Word2VecConfig) -> None:
        self._load_config(config)

        self.cum_table = None

        self.wv = KeyedVectors(self.vector_size)
        self.wv.vectors_lockf = np.ones(1, dtype=REAL)
        self.syn1 = None
        self.syn1neg = None

        self.neg_coef = 0.5
        self.pos_coef = 0.5
        self.neg_count = 0
        self.pos_count = 0

    def _load_config(self, config: Word2VecConfig):
        self.hs = config.hs
        self.cbow_mean = config.cbow_mean
        self.window = config.window
        self.random = np.random.RandomState(seed=config.seed)
        self.sample = config.sample
        self.shrink_windows = config.shrink_windows
        self.negative = config.negative
        self.vector_size = config.vector_size

        self.workers = config.workers
        self.compute_loss = config.compute_loss
        self.running_training_loss = config.running_training_loss


class PLStream(MapFunction):
    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config
        super().__init__()

    def open(self, runtime_context: RuntimeContext):
        self.model = Word2VecModel(self.config)
        self.vocab_storage = VocabStorage()
        self.storage = ModelStorage(
            self.config.vector_size, self.config.layer1_size, self.config.seed)
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.batch_X, self.batch_y = [], []

    def map(self, value):
        start_time = time.time()
        tag, _, X, y = value
        clean_X = clean_sentence(X, self.stemmer)

        batch_size, acc, f1 = 1, 0, 0
        if tag == 'train':
            self.batch_X.append(clean_X)
            self.batch_y.append(y)
            if len(self.batch_X) >= self.config.batch_size:
                self.vocab_storage.set_vocab(self.batch_X)
                self._load_vocab()
                self._load_param()
                self._fit_model()
                acc, f1 = self._eval_model(
                    self.batch_X, self.batch_y, is_train=True)
                batch_size = len(self.batch_X)
                self.batch_X, self.batch_y = [], []
            else:
                tag = self.config.batching_flag
        elif tag == 'eval':
            self.batch_X.append(clean_X)
            self.batch_y.append(y)
            if len(self.batch_X) >= self.config.batch_size:
                acc, f1 = self._eval_model(
                    self.batch_X, self.batch_y, is_train=False)
                batch_size = len(self.batch_X)
                self.batch_X, self.batch_y = [], []
            else:
                tag = self.config.batching_flag

        end_time = time.time()
        duration = end_time - start_time
        # return dumps(f'{tag},{batch_size},{acc},{f1},{start_time},{end_time},{duration}')
        return (tag, batch_size, acc, f1)

    def _load_vocab(self):
        index_to_key, key_to_index = self.vocab_storage.get_index()
        vocab = self.vocab_storage.get_vocab()
        self.model.wv.index_to_key = index_to_key
        self.model.wv.key_to_index = key_to_index

        self.model.wv.expandos["sample_int"] = make_subsample(
            index_to_key, key_to_index, vocab, self.config.sample)

        if self.model.wv.expandos["sample_int"] is None:
            raise ValueError(
                f"{self.vocab_storage.vocab.version, len(index_to_key), len(vocab)}")

        self.model.cum_table = make_cum_table(
            index_to_key, vocab, self.config.domain, self.config.ns_exponent)
        if self.config.hs:
            self.model.wv.expandos["code"], self.model.wv.expandos["point"] = make_huffman_tree(
                index_to_key, key_to_index, vocab)

    def _load_param(self):
        tlen = len(self.model.wv)
        self.model.wv.vectors = self.storage.get_vectors(tlen)
        if self.model.hs:
            self.model.syn1 = self.storage.get_weights(tlen)
        else:
            self.model.syn1neg = self.storage.get_weights(tlen)

    def _fit_model(self):
        work = matutils.zeros_aligned(self.config.layer1_size, dtype=REAL)
        neu1 = matutils.zeros_aligned(self.config.layer1_size, dtype=REAL)

        for cur_epoch in range(self.config.epochs):
            alpha = compute_alpha(
                self.config.min_alpha, self.config.alpha, cur_epoch/self.config.epochs)
            if self.config.sg:
                train_batch_sg(self.model, self.batch_X,
                               alpha, work, self.model.compute_loss)
            else:
                train_batch_cbow(self.model, self.batch_X, alpha,
                                 work, neu1, self.model.compute_loss)

        self.storage.set_vectors(self.model.wv.vectors)
        if self.config.hs:
            self.storage.set_weights(self.model.syn1)
        else:
            self.storage.set_weights(self.model.syn1neg)

    def _predict(self, sentence):
        vector = np.zeros(self.config.vector_size, dtype=REAL)
        counter = 0
        for word in sentence:
            if word in self.model.wv.key_to_index:
                vector += self.model.wv[word]
                counter += 1
        if counter != 0:
            vector = vector / counter

        cos_sim_neg = 0
        for word in self.config.ref_neg:
            if word in self.model.wv.key_to_index:
                cos_sim_neg += cos_similarity(vector, self.model.wv[word])

        cos_sim_pos = 0
        for word in self.config.ref_pos:
            if word in self.model.wv.key_to_index:
                cos_sim_pos += cos_similarity(vector, self.model.wv[word])

        if cos_sim_neg - cos_sim_pos > self.config.confidence:
            return cos_sim_neg - cos_sim_pos, LABEL_NEG
        elif cos_sim_pos - cos_sim_neg > self.config.confidence:
            return cos_sim_pos - cos_sim_neg, LABEL_POS
        elif self.config.ttd:
            if cos_sim_neg * self.model.neg_coef >= cos_sim_pos * self.model.pos_coef:
                return cos_sim_neg - cos_sim_pos, LABEL_NEG
            else:
                return cos_sim_pos - cos_sim_neg, LABEL_POS
        else:
            if cos_sim_neg > cos_sim_pos:
                return cos_sim_neg - cos_sim_pos, LABEL_NEG
            else:
                return cos_sim_pos - cos_sim_neg, LABEL_POS

    def _eval_model(self, batch_X, batch_y, is_train):
        confidences, pred_y = [], []
        for sentence in batch_X:
            confi, y = self._predict(sentence)
            confidences.append(confi)
            pred_y.append(y)

        if is_train:
            self.model.neg_count += pred_y.count(1)
            self.model.pos_count += pred_y.count(0)
            total = self.model.neg_count + self.model.pos_count
            self.model.neg_coef = self.model.neg_count / total
            self.model.pos_coef = self.model.pos_count / total

        acc = accuracy_score(batch_y, pred_y)
        f1 = f1_score(batch_y, pred_y)

        return acc, f1


def run(collection, config: Word2VecConfig, parallelism=1):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    plstream = env.from_collection(collection).name('CollectionSource') \
        .map(PLStream(config)).name('PLStream').set_parallelism(parallelism) \
        .filter(lambda x: x[0] != 'BATCHING') \
        .map(lambda x: f'{x[1]} - {x[2]} - {x[3]}', output_type=Types.STRING()).set_parallelism(1) \
        .add_sink(StreamingFileSink
                  .for_row_format('./output', Encoder.simple_string_encoder())
                  .build())

    # plstream.print()

    env.execute("plstream")


if __name__ == '__main__':
    collection = None

    data_size = 100000
    yelp = read_csv("train.csv")
    yelp.columns = ['label', 'sentence']
    yelp_train_X, yelp_train_y = list(yelp.sentence), list(yelp.label-1)

    collection = []
    # collection = make_input_collection(
    #     'train', yelp_train_X[:data_size], yelp_train_y[:data_size])
    collection = make_input_collection(
        'train', yelp_train_X, yelp_train_y)

    # collection = make_input_collection(
    #     'train', yelp_train_X[:24000], yelp_train_y[:24000]) + make_input_collection(
    #     'eval', yelp_train_X[24000:], yelp_train_y[24000:])

    config = Word2VecConfig(batch_size=250)
    run(collection, config, 8)

# 24000
