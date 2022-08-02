import argparse
import logging
import sys

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext, MapFunction
import pickle
from gensim.models import Word2Vec
import time
from utils import process, split
import pandas as pd
import numpy as np
from pyflink.datastream import CheckpointingMode


def join(x, y):
    logging.warning('x is: ' + str(x))
    logging.warning('y is: ' + str(y))
    return x + y[1]


class Supervised_OSA_inferrence(MapFunction):
    def __init__(self):
        self.model = None
        self.data = []
        self.collector = []
        self.output = []
        self.collector_size = 5

    def open(self, runtime_context: RuntimeContext):
        self.model = Word2Vec.load('word2vec20tokenised.model')

    def map(self, tweet):
        logging.warning(tweet)
        self.data.append(tweet)
        processed_text = process(tweet[2])
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
        self.collector.append([tweet[0], vector_mean, tweet[2], tweet[1]])
        if len(self.collector) >= self.collector_size:
            for e in self.collector:
                self.output.append(e)
            self.collector = []
            return self.output
        else:
            return 'collecting'


class RFClassifier(MapFunction):
    def __init__(self):
        self.model = None
        self.data = []

    def open(self, runtime_context: RuntimeContext):
        file = open('randomforest_classifier', 'rb')
        self.model = pickle.load(file)

    def map(self, ls):
        for i in range(len(ls)):
            self.data.append(ls[i][1])
        confidence = self.model.predict_proba(self.data)
        for i in range(len(ls)):
            if confidence[i][0] >= confidence[i][1]:
                ls[i][1] = confidence[i][0]
                ls[i].insert(2, 0)
            else:
                ls[i][1] = confidence[i][1]
                ls[i].insert(2, 1)

        return ls


def clasifier(ds):
    ds = ds.map(Supervised_OSA_inferrence()).filter(lambda i: i != 'collecting')
    # ds.print()
    # ds.flat_map(split).print() #data size is uneven due to use of collector
    ds = ds.map(RFClassifier()).flat_map(split)
    return ds


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    start_time = time.time()

    # input_path = './yelp_review_polarity_csv/test.csv'
    # if input_path is not None:
    f = pd.read_csv('./exp_train.csv', header=None)  # , encoding='ISO-8859-1'
    f.columns = ["label", "review"]

    f.loc[f['label'] == 1, 'label'] = 0
    f.loc[f['label'] == 2, 'label'] = 1

    true_label = f.label
    yelp_review = f.review
    data_stream = []
    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))
        print((i, int(true_label[i]), yelp_review[i]))
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)

    #  always specify output_type when writing to file

    ds = clasifier(ds).map(lambda x: x[:-1])

    # .key_by(lambda x: x[0])
    ds.print()
    env.execute()
    # supervised_learning(known_args.input, known_args.output)
    logging.info("time taken for execution is: " + str(time.time() - start_time))
