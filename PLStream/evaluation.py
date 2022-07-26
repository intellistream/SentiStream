#!/usr/bin/env python3
import random
import copy
import re
import numpy as np
import argparse

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec

import redis
import pickle
import logging
from newStream import unsupervised_OSA
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
import pandas as pd
import sys
from newStream import unsupervised_stream
from classifier import clasifier
from time import time
from pyflink.datastream import CoMapFunction
from collections import defaultdict

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('plstream.log', mode='w')
formatter = logging.Formatter('PLStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)


def collect(ls, myDict, otherDict, log=False):
    if otherDict[ls[0]] is not None and otherDict[ls[0]] is not False:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[:-1]
            return 'eval'
        else:
            return 'done'
    else:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[:-1]

    logging.warning('mydict in collecting:' + str(myDict.items()))
    return 'collecting'


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

    def evaluation(self, ls, myDict, otherDict):
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
            confidence = self.evaluation(ls, myDict, otherDict)
            if confidence >= 0.5:
                ls[2] = 1
                return ls[2:]
            elif confidence <= -0.5:
                ls[2] = 0
                return ls[2:]
            else:
                return 'low_confidence', confidence, ls[-1]

        else:
            return s

    def map1(self, ls):
        # logging.warning("map1")
        return self.map(ls, self.dict1, self.dict2)

    def map2(self, ls):
        logging.warning("map2")
        return self.map(ls, self.dict2, self.dict1, True)


def evaluation(ds):
    ds1 = unsupervised_stream(ds)
    # ds1.print()
    ds2 = clasifier(ds)
    # ds2.print()
    ds = ds1.connect(ds2) \
        .map(Evaluation()).filter(lambda x: x != 'collecting' and x != 'done') \
        .filter(lambda x: x[0] != 'low_confidence')
    # .key_by(lambda x: x[0])
    return ds


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    start_time = time()

    # input_path = './yelp_review_polarity_csv/test.csv'
    # if input_path is not None:
    f = pd.read_csv('./train.csv', header=None)  # , encoding='ISO-8859-1'
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

    ds = evaluation(ds)

    #  always specify output_type when writing to file
    ds.print()
    env.execute()
    # supervised_learning(known_args.input, known_args.output)
    logging.info("time taken for execution is: " + str(time() - start_time))
