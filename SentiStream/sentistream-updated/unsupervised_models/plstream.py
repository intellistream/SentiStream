# pylint: disable=import-error
# pylint: disable=no-name-in-module
import multiprocessing

from sklearn.metrics import accuracy_score

import config

from unsupervised_models.utils import cos_similarity, text_similarity
from utils import train_word_vector_algo, get_average_word_embeddings, clean_for_wv


class PLStream():
    """
    Online sentiment analysis using PLStream framework.

    Attributes:
        neg_coef (float): Negative coefficient for temporal trend.
        pos_coef (float): Positive coefficient for temporal trend.
        neg_count (int): Number of negative samples seen so far.
        pos_count (int): Number of positive samples seen so far.
        update (bool): Flag determining whether to update word vector model or train from scratch.
        batch_size (int): Number of samples to wait on before processing.
        temporal_trend_detection (bool): If True, perform temporal trend detection.
        confidence (float): Confidence difference to distinguish polarity.
        acc_list(list): Store accuracy of each batch.
        wv_model (class): The word vector model.
        pos_ref (list): List of positive reference words.
        neg_ref (list): List of negative reference words.
        labels (list): Labels of data
        texts (list): Texts/Reviews of data
    """

    def __init__(self, word_vector_algo, vector_size=20, batch_size=250,
                 temporal_trend_detection=True, confidence=0.01):
        """
        Initialize PLStream with hyperparameters.

        Args:
            word_vector_algo (class): Type of word vector algorithm to use (either 'Word2Vec' or
                                    'FastText').
            vector_size (int, optional): Size of word vectors. Defaults to 20.
            batch_size (int, optional): Number of samples to wait on before processing. Defaults 
                                        to 250.
            temporal_trend_detection (bool, optional): If True, perform temporal trend detection.
                                            Defaults to True.
            confidence (float, optional): Confidence difference to distinguish polarity. Defaults 
                                        to 0.5.
        """
        self.neg_coef = 0.5
        self.pos_coef = 0.5
        self.neg_count = 0
        self.pos_count = 0  # watch-out for integer overflow error in horizon.

        self.update = True
        self.batch_size = batch_size
        # self.temporal_trend_detection = temporal_trend_detection
        # self.confidence = confidence

        self.word_vector_algo = word_vector_algo

        self.baseline_acc_list = []
        # self.temporal_list = []
        self.text_similarity_list = []
        self.count = 0
        self.count2 = 0

        # Initialize word vector model.
        # num_workers = int(0.5 * multiprocessing.cpu_count()
        #   )  # Best value for batch of 250.
        # self.wv_model = word_vector_algo(
        #     vector_size=vector_size, workers=num_workers)

        # TODO: TEMP --- WE CAN STILL CALL THIS AS UNSUPERVISED LEARNING??? THIS DOESNT ALTER
        # CLASSIFICATION METHOD. SO....  ---- ALSO CHANGE self.update when changing this ....
        self.wv_model = word_vector_algo.load(config.SSL_WV)

        # Set up positive and negative reference words for trend detection.
        self.pos_ref = {'love', 'best', 'beautiful', 'great',
                        'cool', 'awesome', 'wonderful', 'brilliant', 'excellent', 'fantastic'}
        self.neg_ref = {'bad', 'worst', 'stupid', 'disappointing',
                        'terrible', 'rubbish', 'boring', 'awful', 'unwatchable', 'awkward'}
        if config.STEM:  # TODO: TEMP --- DLT OTHER WHEN FINALIZED.
            # self.pos_ref = {'love', 'best', 'beauti', 'great', 'cool',
            #                 'awesom', 'wonder', 'brilliant', 'excel', 'fantast'}
            # self.neg_ref = {'bad', 'worst', 'stupid', 'disappoint',
            #                 'terribl', 'rubbish', 'bore', 'aw', 'unwatch', 'awkward'}
            self.pos_ref = {'love', 'best', 'beautiful', 'great', 'cool', 'awesome', 'wonderful',
                            'brilliant', 'excellent', 'fantastic', 'super', 'fun', 'masterpiece',
                            'rejoice', 'admire', 'amuse', 'bliss', 'yumm', 'glamour'}
            self.neg_ref = {'bad', 'worst', 'stupid', 'disappointing', 'terrible', 'rubbish',
                            'boring', 'awful', 'unwatchable', 'awkward', 'bullshi', 'fraud',
                            'abuse', 'outrange', 'disgust'}

        self.idx = []
        self.labels = []
        self.texts = []

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
        self.texts.append(clean_for_wv(text))

        # Train model & classify once batch size is reached.
        if len(self.labels) >= self.batch_size:
            train_word_vector_algo(
                self.wv_model, self.texts, config.US_WV, update=self.update)

            self.load_updated_model()

            # Get predictions and confidence scores.
            conf, preds = self.eval_model(self.texts, self.labels)

            # Generate output data
            output = [[i, 'us', c, p, l]
                      for i, c, p, l in zip(self.idx, conf, preds, self.labels)]

            # Clear the lists for the next batch
            self.update = True
            self.idx = []
            self.labels = []
            self.texts = []

            return output
        return config.BATCHING

    # def update_temporal_trend(self, y_preds):
    #     """
    #     Update temporal trend of sentiment analysis based on predictions.

    #     Args:
    #         y_preds (list): Predicted sentiments for current batch.
    #     """
    #     # Calculate positive and negative predictions so far.
    #     for pred in y_preds:
    #         if pred == 1:
    #             self.pos_count += 1
    #         else:
    #             self.neg_count += 1

    #     # Update temporal trend based on predictions.
    #     total = self.neg_count + self.pos_count
    #     self.neg_coef = self.neg_count / total
    #     self.pos_coef = self.pos_count / total

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
        y_tt_preds, y_ts_preds = [], []
        for idx, embeddings in enumerate(doc_embeddings):
            conf, y_pred = self.predict(embeddings)
            # confidence.append(conf)
            y_preds.append(y_pred)

            conf_ts, y_ts_pred = self.predict(
                embeddings, sent_tokens[idx], temp='t')
            confidence.append(conf_ts)
            y_ts_preds.append(y_ts_pred)

            # if y_pred != y_ts_pred and y_pred == self.labels[idx]:
            #     # print(self.labels[idx], sent_tokens[idx], conf, y_pred, conf_ts, y_ts_pred)
            #     self.count += 1

            # if y_pred != y_ts_pred and y_ts_pred == self.labels[idx]:
            #     self.count2 += 1

            # if y_ts_pred == y_pred and y_ts_pred != self.labels[idx]:
            #     print(self.labels[idx], y_ts_pred, y_pred, sent_tokens[idx], conf_ts, conf)

            if y_ts_pred == self.labels[idx]:
                self.count += 1
            else:
                self.count2 += 1

        # self.update_temporal_trend(y_tt_preds)

        self.baseline_acc_list.append(accuracy_score(labels, y_preds))
        self.text_similarity_list.append(accuracy_score(labels, y_ts_preds))
        return confidence, y_preds

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
            for word in self.pos_ref if word in self.wv_model.wv.key_to_index]
        cos_sim_neg = [cos_similarity(
            vector, self.wv_model.wv[word])
            for word in self.neg_ref if word in self.wv_model.wv.key_to_index]

        cos_sim_pos = sum(cos_sim_pos) / len(cos_sim_pos)
        cos_sim_neg = sum(cos_sim_neg) / len(cos_sim_neg)

        if temp == 't':
            # if abs(cos_sim_pos - cos_sim_neg) < 0.01:
            if True and tokens:
                sent_n = [word for word in tokens if not word.startswith('n_')]
                negation = [word for word in tokens if word.startswith('n_')]

                text_sim_pos = [text_similarity(word, sent_n)
                                for word in self.pos_ref]
                text_sim_neg = [text_similarity(word, sent_n)
                                for word in self.neg_ref]

                text_sim_pos += [text_similarity(word, negation, 0.9)
                                 for word in self.neg_ref]
                text_sim_neg += [text_similarity(word, negation, 0.8)
                                 for word in self.pos_ref]

                text_sim_pos = sum(text_sim_pos) / len(tokens)
                text_sim_neg = sum(text_sim_neg) / len(tokens)

                cos_sim_pos += text_sim_pos
                cos_sim_neg += text_sim_neg

        # # Predict polarity based on temporal trend and cosine similarity.
        # if temp == 'tt':
        #     if cos_sim_neg - cos_sim_pos > self.confidence:
        #         return cos_sim_neg - cos_sim_pos, 0
        #     if cos_sim_pos - cos_sim_neg > self.confidence:
        #         return cos_sim_pos - cos_sim_neg, 1
        #     if self.temporal_trend_detection:
        #         if cos_sim_neg * self.neg_coef >= cos_sim_pos * self.pos_coef:
        #             return cos_sim_neg - cos_sim_pos, 0
        #         return cos_sim_pos - cos_sim_neg, 1
        #     if cos_sim_neg > cos_sim_pos:
        #         return cos_sim_neg - cos_sim_pos, 0
        #     return cos_sim_pos - cos_sim_neg, 1

        if cos_sim_neg > cos_sim_pos:
            return (cos_sim_neg + 1)/2, 0
        return (cos_sim_pos + 1)/2, 1
