# import redis
import time
import torch

from pyflink.datastream.functions import MapFunction, RuntimeContext

from utils import load_torch_model, default_model_pretrain, process_text_and_generate_tokens, generate_vector_mean


class Preprocessor(MapFunction):
    def __init__(self, with_accuracy=True):
        """Initializes class with model.

        Parameters:
            with_accuracy (bool, optional): set. Defaults to True.
        """
        self.model = None
        self.collector = []
        self.collector_size = 16
        # self.redis = None
        self.time_threshold = 10 * 60
        self.start_time = time.time()

    def open(self, runtime_context: RuntimeContext):
        """Initialize word2vec model before starting stream/batch processing.

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = default_model_pretrain(
            'w2v.model')
        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    # def load(self):
    #     """
    #     Periodically load updated model
    #     """
    #     if time.time() - self.start_time >= self.time_threshold:
    #         self.start_time = 0
    #         try:
    #             flag = int(self.redis.get('word_vector_update'))
    #             if flag == 1:
    #                 self.model = default_model_pretrain(
    #                     'w2v.model')
    #                 self.redis.set('word_vector_update', int(False))
    #             elif flag is None:
    #                 pass
    #                 # logging.warning(
    #                 #     'word_vector model update flag not set. Train model before running')
    #         except:
    #             raise ConnectionError('Failed to open redis')

    def map(self, tweet):
        """
        Map function preprocess data for classifier.

        Parameters:
            tweet (tuple): tuple of tweet and it's label

        Returns:
            (str or list): list of label and avg word vector of tweet if all data per processing
            unit is collected else, 'collecting'
        """
        # TODO: BATCH PREPROCESSING WILL IMPROVE SPEED ------------------------------------------------------

        processed_text = process_text_and_generate_tokens(tweet[2])
        vector_mean = generate_vector_mean(self.model, processed_text)
        self.collector.append([tweet[0], vector_mean, tweet[2]])

        if len(self.collector) >= self.collector_size:
            output = list(self.collector)
            self.collector = []
            return output
        else:
            return 'collecting'


class Classifier(MapFunction):
    """
    Class for Classifier Placeholder.
    """

    def __init__(self):
        """
        Initializes class with model.
        """
        self.model = None
        self.data = []
        # self.redis = None
        self.time_threshold = 10 * 60
        self.start_time = time.time()

    def open(self, runtime_context: RuntimeContext):
        """Initialize classifier model before starting stream/batch processing.

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = load_torch_model('model.pth')
        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    # def load(self):
    #     """
    #     Periodically load updated model
    #     """
    #     if time.time() - self.start_time >= self.time_threshold:
    #         self.start_time = 0
    #         try:
    #             flag = int(self.redis.get('classifier_update'))
    #             if flag == 1:
    #                 self.model = load_torch_model('model.pth')
    #                 self.redis.set('classifier_update', int(False))
    #             elif flag is None:
    #                 pass
    #                 # logging.warning(
    #                 #     'classifier_model update flag not set. Train model before running')
    #         except:
    #             raise ConnectionError('Failed to open redis')

    def get_confidence(self):
        """
        Predict label and calculate confidence.

        Returns:
            (list): list of float value representing the confidence scores for polarity.
        """

        return self.model(torch.FloatTensor(self.data))

    def map(self, data):
        """
        Map function to predict polarity of datastream.

        Parameters:
            data (list): list of elements from datastreams

        Returns:
            (list): list of predicted outputs in format of [index, prediction confidence, 
            predicted label, review, true label]
        """

        for i in range(len(data)):
            self.data.append(data[i][1])

        with torch.no_grad():
            confidence = self.get_confidence().tolist()
        for i in range(len(data)):
            if confidence[i][0] < 0.5:
                data[i][1] = 1 - confidence[i][0]
                data[i].insert(2, 0)
            else:
                data[i][1] = confidence[i][0]
                data[i].insert(2, 1)

        return data


def classifier(ds):
    """
    Placeholder for Sentiment classifier

    Args:
        ds (datastream): stream of data from source

    Returns:
        datastream: predicted datastream
    """
    ds = ds.map(Preprocessor()).filter(lambda i: i != 'collecting')
    ds = ds.map(Classifier()) \
        .flat_map(lambda x: x)  # flatten

    return ds
