# pylint: disable=import-error
# pylint: disable=no-name-in-module
import math
import numpy as np

from gensim.models import Word2Vec

import config

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

    Constants:
        SIMILARITY_THRESHOLD: Threshold for cosine similarity that new words need to exceed to be 
        added to lexicon.
    """
    SIMILARITY_THRESHOLD = 0.9

    def __init__(self, batch_size=10000, confidence=0.1):
        """
        Initialize PLStream with hyperparameters.

        Args:
            batch_size (int, optional): Number of samples to wait on before processing. Defaults 
                                        to 5000.
            confidence (float, optional): Confidence difference to distinguish polarity. Defaults 
                                        to 0.09.
        """
        self.batch_size = batch_size
        self.confidence = confidence

        self.eval_list = []

        # Load pre-trained word vector model.
        self.wv_model = Word2Vec.load(config.SSL_WV)

        self.id = []
        self.idx = []
        self.labels = []
        self.texts = []

        self.k = 3

        self.pos_ref_vec = None
        self.neg_ref_vec = None
        self.pos_ref_mean = None
        self.neg_ref_mean = None

        # Set up positive and negative reference words for trend detection.
        self.pos_ref = {'love', 'best', 'beautiful', 'great', 'cool', 'awesome', 'wonderful',
                        'brilliant', 'excellent', 'fantastic', 'super', 'fun', 'masterpiece',
                        'rejoice', 'admire', 'amuse', 'bliss', 'yummy', 'glamour'}
        self.neg_ref = {'bad', 'worst', 'stupid', 'disappointing', 'terrible', 'rubbish', 'boring',
                        'awful', 'unwatchable', 'awkward', 'bullshit', 'fraud', 'abuse', 'outrage',
                        'disgust'}

        self.create_lexicon()

    def create_lexicon(self):
        """
        Generate average word embedding for positive and negative reference words.
        """
        # Generate word embeddings for reference words.
        self.neg_ref_vec = np.array([self.wv_model.wv[word]
                                     for word in self.neg_ref
                                     if word in self.wv_model.wv.key_to_index])
        self.pos_ref_vec = np.array([self.wv_model.wv[word]
                                     for word in self.pos_ref
                                     if word in self.wv_model.wv.key_to_index])

        # Calculate average word embeddings.
        self.neg_ref_mean = np.mean(self.neg_ref_vec, axis=0) \
            if self.neg_ref_vec.shape[0] else np.zeros(self.wv_model.vector_size, dtype=np.float32)
        self.pos_ref_mean = np.mean(self.pos_ref_vec, axis=0) \
            if self.pos_ref_vec.shape[0] else np.zeros(self.wv_model.vector_size, dtype=np.float32)

    def update_word_lists(self, sent_vec, update):
        """
        Update positive and negative reference tables from polar words extracted from high
        confident pseudo labels.

        Args:
            sent_vec (list): List of tokens from documents.

        Returns:
            str: 'MODEL_TRAINED' if reference table is updated.
        """

        if update:
            _, texts = zip(*sent_vec)

            # Get all unique words from pseudo labels.
            words = set(
                word for text in texts for word in text if word in self.wv_model.wv.key_to_index)

            # Update reference table with words which have high cosine similarity with
            # words on ref table.
            neg_cos_similarities = [cos_similarity(
                self.wv_model.wv[word], self.neg_ref_mean)
                for word in self.wv_model.wv.key_to_index]
            pos_cos_similarities = [cos_similarity(
                self.wv_model.wv[word], self.pos_ref_mean)
                for word in self.wv_model.wv.key_to_index]

            for i, word in enumerate(words):
                if word not in self.neg_ref and \
                        neg_cos_similarities[i] > PLStream.SIMILARITY_THRESHOLD:
                    self.neg_ref.add(word)
                if word not in self.pos_ref and \
                        pos_cos_similarities[i] > PLStream.SIMILARITY_THRESHOLD:
                    self.pos_ref.add(word)

        # Update average word embedding for pos and neg ref words.
        self.create_lexicon()

    def process_data(self, data):
        """
        Process incoming stream and output polarity of stream data.

        Args:
            data (tuple): Contains index, label and text data.

        Returns:
            tuple or str: 'BATCHING' if collecting data for batch, else, accuracy and f1 score 
                        for current batch's predictions.
        """

        id, idx, label, text = data

        # Append idx, label and preprocessed text to respective lists.
        self.id.append(id)
        self.idx.append(idx)
        self.labels.append(label)
        self.texts.append(text)

        # Train model & classify once batch size is reached.
        if len(self.labels) >= self.batch_size or id == '-1':
            self.texts = clean_for_wv(self.texts)

            train_word_vector_algo(
                self.wv_model, self.texts, config.SSL_WV, update=True, save=False, epochs=20)

            # Get predictions and confidence scores.
            conf, preds = self.eval_model(self.id, self.texts, self.labels)

            # Generate output data
            output = [[i, 'us', c, p, l, id]
                      for i, c, p, l, id in zip(self.idx, conf, preds, self.labels, self.id)]

            # Clear the lists for the next batch
            self.id = []
            self.idx = []
            self.labels = []
            self.texts = []

            return output

    def eval_model(self, id, sent_tokens, labels):
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
        for idx, embeddings in enumerate(doc_embeddings):
            conf, y_pred = self.predict(embeddings, tokens=sent_tokens[idx])
            confidence.append(conf)
            y_preds.append(y_pred)

            self.eval_list.append((id[idx], y_pred, labels[idx]))

        return confidence, y_preds

    def predict(self, vector, tokens=None):
        """
        Predict polarity of text based on lexicon based classification.
        Args:
            vector (list): Tokenized words in a text.
        Returns:
            tuple: Confidence of predicted label and predicted label.
        """
        cos_sim_neg = cos_similarity(vector, self.neg_ref_mean)
        cos_sim_pos = cos_similarity(vector, self.pos_ref_mean)

        # 0.1 best so far
        if abs(cos_sim_neg - cos_sim_pos) < self.confidence and tokens:
            sent_n = []
            negation = []
            for word in tokens:
                if not word.startswith('negation_'):
                    sent_n.append(word)
                else:
                    negation.append(word[9:])

            text_sim_pos = [text_similarity(word, self.pos_ref, 0.8)
                            for word in sent_n] + [text_similarity(word, self.neg_ref, 0.8)
                                                   for word in negation]
            text_sim_neg = [text_similarity(word, self.neg_ref, 0.8)
                            for word in sent_n] + [text_similarity(word, self.pos_ref, 0.8)
                                                   for word in negation]

            cos_sim_pos += 4 * sum(text_sim_pos) / len(tokens)
            cos_sim_neg += 4 * sum(text_sim_neg) / len(tokens)

        if cos_sim_neg > cos_sim_pos:
            return 1 / (1 + math.exp(-self.k * cos_sim_neg)), 0
        return 1 / (1 + math.exp(-self.k * cos_sim_pos)), 1
