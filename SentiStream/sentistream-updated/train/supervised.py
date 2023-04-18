# pylint: disable=import-error
# pylint: disable=no-name-in-module

# import redis
import multiprocessing
import pandas as pd

import config

from semi_supervised_models.ann.trainer import Trainer as ANNTrainer
from semi_supervised_models.han.trainer import Trainer as HANTrainer

from utils import clean_for_wv, tokenize, train_word_vector_algo, get_average_word_embeddings


class TrainModel:
    """
    Train model from scratch and then continously train model when data is available.
    """

    def __init__(self, word_vector_algo, ssl_model, init, nrows=1000, vector_size=20, window=5,
                 min_count=5, pseudo_data_threshold=0.0, acc_threshold=0.9):
        """
        Initialize semi-supervised model training

        Args:
            word_vector_algo (class): Type of word vector algorithm to use (either 'Word2Vec' or
                                    'FastText').
            ssl_model (str): Type of SSL model to use (either 'ANN' or 'HAN').
            init (bool): Flag indicating whether start training from scratch or update model.
            nrows (int, optional): Number of rows to read from CSV for initial training. Defaults
                                    to 1000.
            vector_size (int, optional): Size of word vectors. Defaults to 20.
            window (int, optional): Context window size for training word vectors. Defaults to 5.
            min_count (int, optional): Min frequency of a word to be included in vocab. Defaults
                                        to 5.
            pseudo_data_threshold (float, optional): Threshold for number of pseudo data needed to
                                                    update model. Defaults to 0.0.
            acc_threshold (float, optional): Threshold for max accuracy to not update model.
                                            Defaults to 0.9.
        """

        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.pseudo_data_threshold = pseudo_data_threshold
        self.acc_threshold = acc_threshold
        self.word_vector_algo = word_vector_algo
        self.ssl_model = ssl_model

        # Initialize word vector model and load training data.
        if init:
            workers = int(0.5 * multiprocessing.cpu_count())
            self.wv_model = word_vector_algo(
                vector_size=vector_size, window=window, min_count=min_count, workers=workers)
            # TODO: REMOVE PD IMPORT, HAVE SEPARATE TRAINING DATA
            df = pd.read_csv(config.DATA, names=[
                             'label', 'review'], nrows=nrows)
            df['label'] -= 1

            # Preprocess data for training word vectors.
            self.labels = df.label.tolist()
            self.texts = [tokenize(text) for text in df.review.tolist()]
            self.filtered_tokens = [clean_for_wv(text) for text in self.texts]

            # Train word vector model.
            train_word_vector_algo(
                self.wv_model, self.filtered_tokens, config.SSL_WV, not init)

            # Train classifier.
            self.train_classifier(ssl_model, init)

        # try:
        #     self.redis.set('word_vector_update', int(True))
        #     self.redis.set('classifier_update', int(True))
        # except ConnectionError:
        #     raise ConnectionError('Failed to open redis')

    def train_classifier(self, ssl_model, init, old_embeddings=None):
        """
        Train appropiate classifier and store in locally.

        Args:
            ssl_model (str): Type of SSL model (either 'HAN' or 'ANN').
            init (bool): Flag indicating whether start training from scratch or update model.
            old_embeddings (array-like): Embeddings of word vector before updating (needed to load 
                                        HAN model).
        """
        if ssl_model == 'ANN':
            clf = ANNTrainer(
                get_average_word_embeddings(
                    self.wv_model, self.filtered_tokens),
                self.labels, self.wv_model.vector_size, init, downsample=False)
        else:
            clf = HANTrainer(self.texts, self.labels, self.wv_model.wv.key_to_index, [
                self.wv_model.wv[key] for key in self.wv_model.wv.index_to_key], init,
                old_embeddings=old_embeddings, downsample=False)

        # Fit classifier and save model.
        clf.fit_and_save(config.SSL_CLF)

    def update_model(self, data, acc, pseudo_data_threshold=None, acc_threshold=None):
        """


        Args:
            data (_type_): _description_
            acc (float): Current accuracy of SSL model.
        """

        self.labels, self.texts = zip(*data)
        self.filtered_tokens = [clean_for_wv(text) for text in self.texts]

        # Dynamically update thresholds.
        if pseudo_data_threshold:
            self.pseudo_data_threshold = pseudo_data_threshold
        if acc_threshold:
            self.acc_threshold = acc_threshold

        # If there is too low pseudo data or the accuracy is too high, do not update model.
        if (len(self.labels) < self.pseudo_data_threshold or acc > self.acc_threshold):
            print(f'acc: {acc}, threshold: {acc_threshold}\npseudo_data_size: {len(self.labels)}" \
                    " threshold: {pseudo_data_threshold}')
            return config.SKIPPED

        self.wv_model = self.word_vector_algo.load(config.SSL_WV)

        old_embeddings = [self.wv_model.wv[key]
                          for key in self.wv_model.wv.index_to_key]

        # Train word vector model.
        train_word_vector_algo(
            self.wv_model, self.filtered_tokens, config.SSL_WV, True)

        # Train classifier.
        self.train_classifier(self.ssl_model, False,
                              old_embeddings=old_embeddings)

        return config.FINISHED
