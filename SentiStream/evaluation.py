#!/usr/bin/env python3
import argparse

import numpy as np
import logging

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
import pandas as pd
import sys

from pyflink.datastream.connectors import StreamingFileSink
from sklearn.metrics import accuracy_score

from modified_PLStream import unsupervised_stream
from classifier import clasifier
from time import time
from pyflink.datastream import CoMapFunction
from collections import defaultdict
from pyflink.datastream.connectors import StreamingFileSink
from pyflink.common.serialization import Encoder

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('plstream.log', mode='w')
formatter = logging.Formatter('PLStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect(ls, myDict, otherDict, log=False):
    if otherDict[ls[0]] is not None and otherDict[ls[0]] is not False:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[:-2]
            return 'eval'
        else:
            return 'done'
    else:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[:-2]

    logging.warning('mydict in collecting:' + str(myDict.items()))
    return 'collecting'


def generate_labels_and_confidence(confidence, ls):
    if confidence >= 0.5:
        ls[2] = 1
        return ls[2:]
    elif confidence <= -0.5:
        ls[2] = 0
        return ls[2:]
    else:
        if confidence < 0:
            ls[2] = 0
        else:
            ls[2] = 1
        # returns tag,confidence,label,tweet
        return ['low_confidence', confidence, *ls[2:]]


class Evaluation(CoMapFunction):
    def __init__(self):
        self.dict1 = defaultdict(lambda: None)  # for first stream
        self.dict2 = defaultdict(lambda: None)  # for second stream
        self.confidence = 0.5
        self.PLSTREAM_ACC = 0.75  # based on test results
        self.RF_CLASIFIER_ACC = 0.87  # based on test results
        # weights
        self.w1 = self.PLSTREAM_ACC / (self.PLSTREAM_ACC + self.RF_CLASIFIER_ACC)
        self.w2 = self.RF_CLASIFIER_ACC / (self.PLSTREAM_ACC + self.RF_CLASIFIER_ACC)

    def polarity(self, label):
        if label == 1:
            return 1
        else:
            return -1

    def evaluate(self, ls, myDict, otherDict):
        logging.warning('dict1:' + str(self.dict1.items()))
        logging.warning('dict2:' + str(self.dict2.items()))
        logging.warning(ls)
        # labels
        l1 = self.dict1[ls[0]][2]
        l2 = self.dict2[ls[0]][2]

        # confidence
        con1 = self.dict1[ls[0]][1]
        con2 = self.dict2[ls[0]][1]

        # polarity
        sign1 = self.polarity(l1)
        sign2 = self.polarity(l2)

        myDict[ls[0]] = False
        otherDict[ls[0]] = False

        return self.w1 * con1 * sign1 + self.w2 * con2 * sign2

    def map(self, ls, myDict, otherDict, log=False):
        if log == True:
            logging.warning("in map: " + str(ls))
            logging.warning(self.dict1.items())
            logging.warning(self.dict2.items())
        s = collect(ls, myDict, otherDict, log)
        if s == 'eval':
            confidence = self.evaluate(ls, myDict, otherDict)
            return generate_labels_and_confidence(confidence, ls)
        else:
            return s

    def map1(self, ls):
        logging.warning("map1")
        # logging.warning(ls)
        return self.map(ls, self.dict1, self.dict2)

    def map2(self, ls):
        logging.warning("map2")
        return self.map(ls, self.dict2, self.dict1, True)


def merged_stream(ds1, ds2):
    # ds2.print()
    ds = ds1.connect(ds2) \
        .map(Evaluation()).filter(lambda x: x != 'collecting' and x != 'done') \
        .filter(lambda x: x[0] != 'low_confidence')

    # .key_by(lambda x: x[0])
    return ds


def generate_new_label(ds, ds_print=None):
    ds = ds.map(lambda x: x[:-1])
    if not ds_print:
        ds = ds.map(lambda x: str(x[:-1])).add_sink(StreamingFileSink  # .set_parallelism(2)
                                                    .for_row_format('./evaluation', Encoder.simple_string_encoder())
                                                    .build())
    return ds


def calculate_accuracy(ds, ds_print=None):
    data = ds.execute_and_collect()
    true_label = []
    predicted_label = []
    for results in data:
        true_label.append(results[-1])
        predicted_label.append(results[0])
        if ds_print == 'PRINT':
            print(results)
    return accuracy_score(true_label, predicted_label)


if __name__ == '__main__':
    mode = None
    ds_print = None
    parser = argparse.ArgumentParser(description='Run evaluation in two modes, labels and accuracy. Accuracy mode is\
     default')
    parser.add_argument('-l', dest='mode', action='store_const', default='ACC', const='LABEL',
                        help='Generate label(default: print accuracy)')
    parser.add_argument('-p', dest='ds_print', action='store_const', default='PRINT', const=None,
                        help='Generate label(default: print accuracy)')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    start_time = time()

    # input_path = './yelp_review_polarity_csv/test.csv'
    # if input_path is not None:
    f = pd.read_csv('./exp_train.csv', header=None)  # , encoding='ISO-8859-1'
    # f = pd.read_csv('./exp_train.csv', header=None)  # , encoding='ISO-8859-1'
    f.columns = ["label", "review"]
    test_N = 100
    f.loc[f['label'] == 1, 'label'] = 0
    f.loc[f['label'] == 2, 'label'] = 1

    true_label = f.label[:test_N]
    yelp_review = f.review[:test_N]
    data_stream = []
    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))

    print('Coming Stream is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)

    ds1 = unsupervised_stream(ds)
    # ds1.print()
    ds2 = clasifier(ds)
    # ds2.print()
    ds = merged_stream(ds1, ds2)
    if mode == 'LABEL':
        ds = generate_new_label(ds,ds_print)
        ds.print()
        env.execute()
    else:
        print(calculate_accuracy(ds,ds_print))

    #  always specify output_type when writing to file
    logging.info("time taken for execution is: " + str(time() - start_time))
    # supervised_learning(known_args.input, known_args.output)
