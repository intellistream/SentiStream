# pylint: disable=import-error
# pylint: disable=no-name-in-module

# import redis
import multiprocessing
import pandas as pd

from semi_supervised_models.ann.trainer import Trainer as ANNTrainer
from semi_supervised_models.han.trainer import Trainer as HANTrainer

from utils import (load_word_vector_model, load_pseudo_data, clean_for_wv,
                   tokenize, train_word_vector_algo, get_average_word_embeddings)


PSEUDO_DATA_FOLDER = './senti_output'


class TrainModel:
    """
    Train model from scratch and then continously train model when data is available.
    """

    def __init__(self, word_vector_algo, ssl_model, init, nrows=1000, vector_size=20, window=5,
                 min_count=5, acc=0.0, pseudo_data_threshold=0.0, acc_threshold=0.9):
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
            acc (float, optional): Current accuracy of SSL model. Defaults to 0.0.
            pseudo_data_threshold (float, optional): Threshold for number of pseudo data needed to
                                                    update model. Defaults to 0.0.
            acc_threshold (float, optional): Threshold for max accuracy to not update model.
                                            Defaults to 0.9.
        """

        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

        # Initialize word vector model and load training data.
        if init:
            workers = int(0.8 * multiprocessing.cpu_count())
            self.wv_model = word_vector_algo(
                vector_size=vector_size, window=window, min_count=min_count, workers=workers)
            df = pd.read_csv('train.csv', names=[
                             'label', 'review'], nrows=nrows)
            df['label'] -= 1
        else:
            # Load pseudo data.
            df = load_pseudo_data(PSEUDO_DATA_FOLDER)
            # If there is too low pseudo data or the accuracy is too high, do not update model.
            if (len(df) < pseudo_data_threshold or acc > acc_threshold):
                print(f'acc: {acc}, threshold: {acc_threshold}\npseudo_data_size: {len(df)}" \
                        " threshold: {pseudo_data_threshold}')
                return
            self.wv_model = load_word_vector_model(
                word_vector_algo, 'ssl-wv.model')

        # Preprocess data for training word vectors.
        self.labels = df.label.tolist()
        self.texts = [tokenize(text) for text in df.review.tolist()]
        self.filtered_tokens = [clean_for_wv(text) for text in self.texts]

        # Train word vector model.
        train_word_vector_algo(
            self.wv_model, self.filtered_tokens, 'ssl-wv.model', not init)

        # Train classifier.
        self.train_classifier(ssl_model, init)

        # try:
        #     self.redis.set('word_vector_update', int(True))
        #     self.redis.set('classifier_update', int(True))
        # except ConnectionError:
        #     raise ConnectionError('Failed to open redis')

    def train_classifier(self, ssl_model, init):
        """
        Train appropiate classifier and store in locally.

        Args:
            ssl_model (str): Type of SSL model (either 'HAN' or 'ANN').
            init (bool): Flag indicating whether start training from scratch or update model.
        """
        if ssl_model == 'ANN':
            clf = ANNTrainer(
                [get_average_word_embeddings(self.wv_model, tokens)
                 for tokens in self.filtered_tokens],
                self.labels, self.wv_model.vector_size, init, downsample=False)
        else:
            clf = HANTrainer(self.texts, self.labels, self.wv_model.wv.key_to_index, [
                self.wv_model.wv[key] for key in self.wv_model.wv.index_to_key], init,
                downsample=False)

        # Fit classifier and save model.
        clf.fit_and_save('ssl-clf.pth')

        print('FINISHED TRAINING')
