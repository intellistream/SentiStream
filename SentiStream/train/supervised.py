# pylint: disable=import-error
# pylint: disable=no-name-in-module
import config

from gensim.models import Word2Vec

from semi_supervised_models.trainer import Trainer
from utils import clean_for_wv, train_word_vector_algo, downsampling


class TrainModel:
    """
    Train model from scratch and then continously train model when data is available.
    """

    def __init__(self, init, data=None, vector_size=20, window=5,
                 min_count=5, test_size=0.2, batch_size=512, lr=0.002):
        """
        Initialize semi-supervised model training

        Args:
            ssl_model (str): Type of SSL model to use (either 'ANN' or 'HAN').
            init (bool): Flag indicating whether start training from scratch or update model.
            vector_size (int, optional): Size of word vectors. Defaults to 20.
            window (int, optional): Context window size for training word vectors. Defaults to 5.
            min_count (int, optional): Min frequency of a word to be included in vocab. Defaults
                                        to 5.
        """
        self.test_size = test_size

        self.batch_size = batch_size
        self.lr = lr

        self.labels = []
        self.texts = []

        # Initialize word vector model and load training data.
        if init:
            self.wv_model = Word2Vec(
                vector_size=vector_size, window=window, min_count=min_count, workers=8)

            self.labels, self.texts = zip(*data)

            # Preprocess data for training word vectors.
            self.filtered_tokens = clean_for_wv(self.texts)

            # Train word vector model.
            train_word_vector_algo(
                self.wv_model, self.filtered_tokens, config.SSL_WV,
                update=not init, min_count=min_count)

            # Downsample to balance classes.
            self.labels, self.texts = downsampling(self.labels, self.texts)

            # Train classifier.
            self.train_classifier(init)

    def train_classifier(self, init, old_embeddings=None):
        """
        Train appropiate classifier and store in locally.

        Args:
            init (bool): Flag indicating whether start training from scratch or update model.
            old_embeddings (array-like): Embeddings of word vector before updating (needed to load 
                                        HAN model).
        """
        clf = Trainer(self.texts, self.labels, self.wv_model.wv.key_to_index, [
            self.wv_model.wv[key] for key in self.wv_model.wv.index_to_key], init,
            old_embeddings=old_embeddings, test_size=self.test_size, batch_size=self.batch_size,
            learning_rate=self.lr)

        # Fit classifier and save model.
        clf.fit_and_save(config.SSL_CLF)

    def update_model(self, data, pseudo_data_threshold=None):
        """
        Update pretrained model using incremental learning technique.

        Args:
            data (list): Tuples containing label and processed text.
            pseudo_data_threshold (float, optional): Threshold for number of pseudo data needed to
                                                    update model. Defaults to None.

        Returns:
            str: 'FINISHED' if model is successfully updated, 'SKIPPED' if current batch doesn't 
                meet requirements to be trained.
        """
        self.labels, self.texts = zip(*data)

        # Downsample to balance classes.
        self.labels, self.texts = downsampling(self.labels, self.texts)

        # If there is too low pseudo data, do not update model.
        if len(self.labels) < pseudo_data_threshold:
            return config.SKIPPED

        self.filtered_tokens = clean_for_wv(self.texts)

        self.wv_model = Word2Vec.load(config.SSL_WV)
        old_embeddings = [self.wv_model.wv[key]
                          for key in self.wv_model.wv.index_to_key]

        # Train word vector model.
        train_word_vector_algo(
            self.wv_model, self.filtered_tokens, config.SSL_WV, True, epochs=20)

        # Train classifier.
        self.train_classifier(False, old_embeddings=old_embeddings)

        return config.FINISHED
