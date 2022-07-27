#!/usr/bin/env python3
import numpy as np
import redis
import pickle
import logging

from sklearn.metrics import accuracy_score

from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode
import sys

from utils import process, load_and_augment_data

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('plstream.log', mode='w')
formatter = logging.Formatter('PLStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec
from os import walk
import os

pseudo_data_size = 0
test_data_size = 0


class Supervised_OSA_inferrence(MapFunction):
    def __init__(self):
        self.model = None
        self.collector = []
        self.output = []
        self.collector_size = int(test_data_size / parallelism)
        # logging.warning("pseudo_data_size: " + str(pseudo_data_size))

    def open(self, runtime_context: RuntimeContext):
        self.model = Word2Vec.load('word2vec20tokenised.model')

    def map(self, tweet):
        # logging.warning(tweet)
        processed_text = process(tweet[1])
        word_vector = []
        for token in processed_text:
            try:
                word_vector.append(self.model.wv[token])
            except:
                pass
        if len(word_vector) == 0:
            vector_mean = np.zeros(self.model.vector_size)
        else:
            vector_mean = (np.mean(word_vector, axis=0)).tolist()
        self.collector.append([tweet[0], vector_mean])
        logging.warning(self.collector_size)
        logging.warning(len(self.collector))
        if len(self.collector) >= self.collector_size:
            logging.warning("in condintion:" + str(len(self.collector)))
            for e in self.collector:
                self.output.append(e)
            self.collector = []
            # logging.warning(self.output)
            return self.output
        else:
            return 'collecting'


class RFClassifier(MapFunction):
    def __init__(self):
        self.model = None
        self.vectors = []
        self.labels = []
        self.data = []
        self.redis_param = None

    def open(self, runtime_context: RuntimeContext):
        file = open('randomforest_classifier', 'rb')
        self.model = pickle.load(file)

    def map(self, ls):
        for i in range(len(ls)):
            self.labels.append(ls[i][0])
            self.data.append(ls[i][1])

        # logging.warning(self.labels)
        # logging.warning(self.data)

        predictions = self.model.predict(self.data)
        accuracy = accuracy_score(self.labels, predictions)
        return 1, accuracy


def batch_inference(ds, supervised_parallelism=1, clasifier_parallelism=1):
    global parallelism
    redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    parallelism = supervised_parallelism
    ds = ds.map(Supervised_OSA_inferrence()).set_parallelism(supervised_parallelism).filter(lambda i: i != 'collecting')
    # ds.flat_map(split).print() #data size is uneven due to use of collector
    ds = ds.map(RFClassifier()).set_parallelism(clasifier_parallelism) \
        .key_by(lambda x: x[0]).reduce(lambda x, y: (1, (x[1] + y[1]) / 2))

    with ds.execute_and_collect() as results:
        for accuracy in results:
            redis_param.set('batch_inference_accuracy', accuracy[1].item())
            print(type(accuracy[1].item()))
    return accuracy[1]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # data source
    pseudo_data_folder = './senti_output'
    test_data_file = './exp_test.csv'

    # data sets
    pseudo_data_size, test_df = load_and_augment_data(pseudo_data_folder, test_data_file)
    test_data_size = len(test_df)
    # test_N = 100
    # true_label = df.label[:test_N]
    # yelp_review = df.review[:test_N]

    true_label = test_df.label
    yelp_review = test_df.review
    data_stream = []
    for i in range(len(yelp_review)):
        data_stream.append((int(true_label[i]), yelp_review[i]))

    print('Coming Stream is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)
    accuracy = batch_inference(ds)
    print(accuracy)
    # env.execute()
