# pylint: disable=import-error
# pylint: disable=no-name-in-module
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import config
import numpy as np

from unsupervised_models.utils import cos_similarity, text_similarity
from utils import train_word_vector_algo, get_average_word_embeddings, clean_for_wv


class PLStream():
    """
    Online sentiment analysis using PLStream framework.

    Attributes:
        update (bool): Flag determining whether to update word vector model or train from scratch.
        batch_size (int): Number of samples to wait on before processing.
        confidence (float): Confidence difference to distinguish polarity.
        acc_list(list): Store accuracy of each batch.
        wv_model (class): The word vector model.
        labels (list): Labels of data
        texts (list): Texts/Reviews of data
    """

    def __init__(self, word_vector_algo, batch_size=10000, confidence=0.1):
        """
        Initialize PLStream with hyperparameters.

        Args:
            word_vector_algo (class): Type of word vector algorithm to use (either 'Word2Vec' or
                                    'FastText').
            batch_size (int, optional): Number of samples to wait on before processing. Defaults 
                                        to 5000.
            confidence (float, optional): Confidence difference to distinguish polarity. Defaults 
                                        to 0.09.
        """
        self.batch_size = batch_size
        self.word_vector_algo = word_vector_algo
        self.confidence = confidence

        self.acc_list = []
        self.ts_list = []
        self.l_list = []
        self.lt_list = []

        self.count = 0
        self.count2 = 0

        # Load pre-trained word vector model.
        self.wv_model = word_vector_algo.load(config.SSL_WV)

        self.idx = []
        self.labels = []
        self.texts = []

        self.lexicon_size = 10

        self.pos_ref_vec = None
        self.neg_ref_vec = None
        self.pos_ref_mean = None
        self.neg_ref_mean = None

        self.create_lexicon()
        self.create_lexicon_temp()

    def create_lexicon(self):
        """
        Generate average word embedding for positive and negative reference words.
        """
        # Generate word embeddings for reference words.
        self.neg_ref_vec = np.array([self.wv_model.wv[word]
                                     for word in config.NEG_REF
                                     if word in self.wv_model.wv.key_to_index])
        self.pos_ref_vec = np.array([self.wv_model.wv[word]
                                     for word in config.POS_REF
                                     if word in self.wv_model.wv.key_to_index])

        # Calculate average word embeddings.
        self.neg_ref_mean = np.mean(self.neg_ref_vec, axis=0)
        self.pos_ref_mean = np.mean(self.pos_ref_vec, axis=0)

    def create_lexicon_temp(self):
        """
        Generate average word embedding for positive and negative reference words.
        """
        # Generate word embeddings for reference words.
        self.neg_ref_vec_temp = np.array([self.wv_model.wv[word]
                                          for word in config.NEG_REF
                                          if word in self.wv_model.wv.key_to_index])
        self.pos_ref_vec_temp = np.array([self.wv_model.wv[word]
                                          for word in config.POS_REF
                                          if word in self.wv_model.wv.key_to_index])

        # Calculate average word embeddings.
        self.neg_ref_mean_temp = np.mean(self.neg_ref_vec_temp, axis=0)
        self.pos_ref_mean_temp = np.mean(self.pos_ref_vec_temp, axis=0)

    def update_word_lists(self, sentence_vectors, temp=None):
        """
        Update positive and negative reference tables from polar words extracted from high
        confident pseudo labels.

        Args:
            sentence_vectors (list): List of tokens from documents.

        Returns:
            str: 'MODEL_TRAINED' if reference table is updated.
        """

        labels, texts = zip(*sentence_vectors)

        if temp == 't':
            # Get all unique words from pseudo labels.
            words = set(
                word for text in texts for word in text if word in self.wv_model.wv.key_to_index)

            # Update reference table with words which have high cosine similarity with
            # words on ref table.
            for word in words:
                if word not in config.NEG_REF and \
                        cos_similarity(self.wv_model.wv[word], self.neg_ref_mean_temp) > 0.9:
                    config.NEG_REF.add(word)
                if word not in config.POS_REF and \
                        cos_similarity(self.wv_model.wv[word], self.pos_ref_mean_temp) > 0.9:
                    config.POS_REF.add(word)

            # Update average word embedding for pos and neg ref words.
            self.create_lexicon_temp()
        else:
            self.create_lexicon()
            sen_embeddings = get_average_word_embeddings(
                self.wv_model, texts)
            # Add the sentence vectors to the combined word vectors
            for sent_vec, label in zip(sen_embeddings, labels):
                if label == 0:
                    self.neg_ref_vec = np.vstack([self.neg_ref_vec, sent_vec])
                elif label == 1:
                    self.pos_ref_vec = np.vstack([self.pos_ref_vec, sent_vec])

            # Check if the threshold is exceeded
            if self.neg_ref_vec.shape[0] + self.pos_ref_vec.shape[0] > self.lexicon_size:
                # Determine the number of clusters for negative and positive sentiments
                if self.neg_ref_vec.shape[0] > self.lexicon_size/2:
                    n_clusters_negative = self.lexicon_size//2
                if self.pos_ref_vec.shape[0] > self.lexicon_size/2:
                    n_clusters_positive = self.lexicon_size//2

                # Perform KMeans clustering for negative sentiment
                kmeans_negative = KMeans(
                    n_clusters=n_clusters_negative, n_init=10)
                kmeans_negative.fit(self.neg_ref_vec)
                self.neg_ref_vec = kmeans_negative.cluster_centers_

                # Perform KMeans clustering for positive sentiment
                kmeans_positive = KMeans(
                    n_clusters=n_clusters_positive, n_init=10)
                kmeans_positive.fit(self.pos_ref_vec)
                self.pos_ref_vec = kmeans_positive.cluster_centers_
                # get means
                self.neg_ref_mean = self.neg_ref_vec.sum(axis=0)
                self.neg_ref_mean = self.neg_ref_mean / \
                    self.neg_ref_vec.shape[0]
                self.pos_ref_mean = self.pos_ref_vec.sum(axis=0)
                self.pos_ref_mean = self.pos_ref_mean / \
                    self.pos_ref_vec.shape[0]

        return config.FINISHED

    # TODO: USE WHEN MULTIPROCESSING
    def load_updated_model(self):
        """
        Load updated model from local.
        """
        self.wv_model = self.word_vector_algo.load(config.US_WV)

    def process_data(self, data):
        """
        Process incoming stream and output polarity of stream data.

        Args:
            data (tuple): Contains index, label and text data.

        Returns:
            tuple or str: 'BATCHING' if collecting data for batch, else, accuracy and f1 score 
                        for current batch's predictions.
        """

        idx, label, text = data

        # Append idx, label and preprocessed text to respective lists.
        self.idx.append(idx)
        self.labels.append(label)
        self.texts.append(text)

        # Train model & classify once batch size is reached.
        if len(self.labels) >= self.batch_size:
            self.texts = clean_for_wv(self.texts)

            train_word_vector_algo(
                self.wv_model, self.texts, config.US_WV, update=True, save=False, epochs=10)

            # Get predictions and confidence scores.
            conf, preds = self.eval_model(self.texts, self.labels)

            # Generate output data
            output = [[i, 'us', c, p, l]
                      for i, c, p, l in zip(self.idx, conf, preds, self.labels)]

            # Clear the lists for the next batch
            self.idx = []
            self.labels = []
            self.texts = []

            return output
        return config.BATCHING

    def eval_model(self, sent_tokens, labels):
        """
        Evaluate model on current batch

        Args:
            sent_tokens (list): Tokenized texts.
            labels (list): Sentiment labels.

        Returns:
            tuple: Accuracy and F1 score of model on current batch.
        """

        # Calculate average word embeddings for text.
        doc_embeddings = get_average_word_embeddings(
            self.wv_model, sent_tokens)

        confidence, y_preds = [], []
        y_ts_preds, y_l_preds, y_lt_preds = [], [], []
        for idx, embeddings in enumerate(doc_embeddings):
            # conf, y_pred = self.predict_t(embeddings, tokens=sent_tokens[idx], temp='t', tt=0)
            # y_lt_preds.append(y_pred)

            # if y_pred != labels[idx]:
            #     print(y_pred, labels[idx], sent_tokens[idx])

            # conf, y_pred = self.predict_t(embeddings, tokens=sent_tokens[idx])
            # y_ts_preds.append(y_pred)

            conf, y_pred = self.predict_t(
                embeddings, tokens=sent_tokens[idx], temp='t')
            y_l_preds.append(y_pred)

            conf, y_pred = self.predict_t(
                embeddings, tokens=sent_tokens[idx], temp='t', tt='t')
            confidence.append(conf)
            y_preds.append(y_pred)

        self.l_list.append(accuracy_score(labels, y_l_preds))
        # self.lt_list.append(accuracy_score(labels, y_lt_preds))
        # self.ts_list.append(accuracy_score(labels, y_ts_preds))
        self.acc_list.append(accuracy_score(labels, y_preds))
        return confidence, y_preds

    def predict_t(self, vector, tokens=None, temp=None, tt=None):
        """
        Predict polarity of text based using PLStream.
        Args:
            vector (list): Tokenized words in a text.
        Returns:
            tuple: Confidence of predicted label and predicted label.
        """
        if tt == 't':
            cos_sim_neg = cos_similarity(vector, self.neg_ref_mean_temp)
            cos_sim_pos = cos_similarity(vector, self.pos_ref_mean_temp)
        else:
            cos_sim_neg = cos_similarity(vector, self.neg_ref_mean)
            cos_sim_pos = cos_similarity(vector, self.pos_ref_mean)

        # 0.09 best so far
        if temp == 't':
            if abs(cos_sim_neg - cos_sim_pos) < self.confidence and tokens:
                # if tokens:
                sent_n = [
                    word for word in tokens if not word.startswith('negation_')]
                negation = [word[9:]
                            for word in tokens if word.startswith('negation_')]

                text_sim_pos = [text_similarity(word, sent_n, 0.8)
                                for word in config.POS_REF] + [text_similarity(word, negation, 0.8)
                                                               for word in config.NEG_REF]
                text_sim_neg = [text_similarity(word, sent_n, 0.8)
                                for word in config.NEG_REF] + [text_similarity(word, negation, 0.8)
                                                               for word in config.POS_REF]

                cos_sim_pos += sum(text_sim_pos) / len(tokens)
                cos_sim_neg += sum(text_sim_neg) / len(tokens)

        if cos_sim_neg > cos_sim_pos:
            return (cos_sim_neg + 1)/2, 0
        return (cos_sim_pos + 1)/2, 1

    def predict(self, vector, tokens=None, temp=None):
        """
        Predict polarity of text based using PLStream.

        Args:
            vector (list): Tokenized words in a text.

        Returns:
            tuple: Confidence of predicted label and predicted label.
        """

        cos_sim_pos = [cos_similarity(
            vector, self.wv_model.wv[word])
            for word in config.POS_REF if word in self.wv_model.wv.key_to_index]
        cos_sim_neg = [cos_similarity(
            vector, self.wv_model.wv[word])
            for word in config.NEG_REF if word in self.wv_model.wv.key_to_index]

        cos_sim_pos = sum(cos_sim_pos) / len(cos_sim_pos)
        cos_sim_neg = sum(cos_sim_neg) / len(cos_sim_neg)

        # 0.09 best so far
        if temp == 't':
            if abs(cos_sim_neg - cos_sim_pos) < self.confidence and tokens:
                sent_n = [
                    word for word in tokens if not word.startswith('negation_')]
                negation = [word[9:]
                            for word in tokens if word.startswith('negation_')]

                text_sim_pos = [text_similarity(word, sent_n, 0.9)
                                for word in config.POS_REF] + [text_similarity(word, negation, 0.8)
                                                               for word in config.NEG_REF]
                text_sim_neg = [text_similarity(word, sent_n, 0.9)
                                for word in config.NEG_REF] + [text_similarity(word, negation, 0.8)
                                                               for word in config.POS_REF]

                cos_sim_pos += sum(text_sim_pos) / len(tokens)
                cos_sim_neg += sum(text_sim_neg) / len(tokens)

        if cos_sim_neg > cos_sim_pos:
            return (cos_sim_neg + 1)/2, 0
        return (cos_sim_pos + 1)/2, 1
