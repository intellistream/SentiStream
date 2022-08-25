import pickle
import sys

import redis
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.functions import RuntimeContext, MapFunction
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import load_and_augment_data, pre_process, default_model_pretrain, train_word2vec, generate_vector_mean

# global variables
PSEUDO_DATA_COLLECTION_THRESHOLD = 0
ACCURACY_THRESHOLD = 0.9
parallelism = 1
train_data_size = 0


class Supervised_OSA(MapFunction):
    def __init__(self, train_data_size):
        self.model = None
        self.sentences = []
        self.labels = []
        self.output = []
        self.collection_threshold = train_data_size
        self.redis = None  # do not set redis variable here it gives error
        # logging.warning("pseudo_data_size: " + str(pseudo_data_size))

    def open(self, runtime_context: RuntimeContext):
        """
        upload model here  in runtime
        """
        self.model = default_model_pretrain("PLS_c10.model")  # change to your model
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def initialise_classifier_and_fit(self, mean_vectors, classifier=RandomForestClassifier):
        """
        :param mean_vectors: list of mean_vectors of all the train sentences
        :param classifier: choice of classifier
        :return: a fitted classifier
        """
        clf = classifier()
        self.classifier_fit(mean_vectors, clf.fit)  # change to another fit function from your model if applicable
        return clf

    def classifier_fit(self, mean_vectors, func):
        """

        :param mean_vectors: list of mean_vectors of all the train sentences
        :param func: a function to fit classifier
        :return: None, only fits the classifier
        """
        func(mean_vectors, self.labels)

    def map(self, tweet):
        # logging.warning(tweet)

        self.sentences.append(tweet[1])
        self.labels.append(tweet[0])
        if len(self.labels) >= self.collection_threshold:
            self.train_wordvector_model()  # pretraining model

            mean_vectors = []
            for sentence in self.sentences:
                mean_vectors.append(generate_vector_mean(self.model, sentence))  # change to custom vector mean function

            clf = self.initialise_classifier_and_fit(mean_vectors)  # change to your model

            filename = 'supervised.model'
            pickle.dump(clf, open(filename, 'wb'))

            try:
                self.redis.set('word_vector_update', int(True))
                self.redis.set('classifier_update', int(True))
            except ConnectionError:
                raise ConnectionError('Failed to open redis')

            return "finished training"
        else:
            return 'collecting'

    def train_wordvector_model(self, func=train_word2vec):
        """

        :param func: a function that expects a pretraining model and the sentences to train
        :return: None. it trains self.model of this object
        """
        func(self.model, self.sentences)


def supervised_model(data_process_parallelism, train_df, train_data_size, pseudo_data_size,
                     PSEUDO_DATA_COLLECTION_THRESHOLD,
                     accuracy,
                     ACCURACY_THRESHOLD,
                     init=False):
    if init or (pseudo_data_size > PSEUDO_DATA_COLLECTION_THRESHOLD and accuracy < ACCURACY_THRESHOLD):

        # data preparation
        true_label = train_df.label
        yelp_review = train_df.review
        data_stream = []
        for i in range(len(yelp_review)):
            data_stream.append((int(true_label[i]), yelp_review[i]))

        # stream environment setup
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_runtime_mode(RuntimeExecutionMode.BATCH)
        env.set_parallelism(1)
        env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

        print('Coming Stream is ready...')
        print('===============================')

        # data stream pipline
        ds = env.from_collection(collection=data_stream)
        ds = ds.map(pre_process)  # change to your pre_processing function,
        ds = ds.set_parallelism(data_process_parallelism).map(Supervised_OSA(train_data_size))
        ds = ds.filter(lambda x: x != 'collecting')
        # ds = batch_inference(ds)

        ds.print()

        env.execute()
    else:
        print("accuracy below threshold: " + str(accuracy < ACCURACY_THRESHOLD))
        print("pseudo data above threshold: " + str(pseudo_data_size > PSEUDO_DATA_COLLECTION_THRESHOLD))
        print("Too soon to update model")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # data source
    pseudo_data_folder = './senti_output'
    train_data_file = './exp_train.csv'

    # data sets
    pseudo_data_size, train_df = load_and_augment_data(pseudo_data_folder, train_data_file)

    train_data_size = len(train_df)

    redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    # accuracy = float(redis_param.get('batch_inference_accuracy').decode())
    accuracy = 0.4
    supervised_model(parallelism, train_df, train_data_size, pseudo_data_size, PSEUDO_DATA_COLLECTION_THRESHOLD,
                     accuracy,
                     ACCURACY_THRESHOLD)
