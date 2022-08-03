#!/usr/bin/env python3
import argparse

import numpy as np
import logging

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
import pandas as pd
import sys

from sklearn.metrics import accuracy_score

from modified_PLStream import unsupervised_stream
from classifier import classifier
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


def default_confidence(ls, myDict, otherDict):
    # logging.warning('dict1:' + str(myDict.items()))
    # logging.warning('dict2:' + str(myDict.items()))
    logging.warning(ls)
    # labels
    l1 = myDict[ls[0]][1]
    l2 = otherDict[ls[0]][1]

    # confidence
    con1 = myDict[ls[0]][0]
    con2 = otherDict[ls[0]][0]

    # polarity
    sign1 = polarity(l1)
    sign2 = polarity(l2)

    myDict[ls[0]] = False
    otherDict[ls[0]] = False

    return Evaluation.w1 * con1 * sign1 + Evaluation.w2 * con2 * sign2


def collect(ls, myDict, otherDict, log=False):
    if otherDict[ls[0]] is not None and otherDict[ls[0]] is not False:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[1:-2] + [ls[-1]]
            return 'eval'
        else:
            return 'done'
    else:
        if myDict[ls[0]] is None:
            myDict[ls[0]] = ls[1:-2] + [ls[-1]]

    logging.warning('mydict in collecting:' + str(myDict.items()))
    return 'collecting'


def generate_label_from_confidence(confidence, ls):
    if confidence >= 0.5:
        ls[2] = 1
        # logging.warning(ls)
        # logging.warning("high positive confidence")
        return ls[2:]
    elif confidence <= -0.5:
        ls[2] = 0
        # logging.warning("high negative confidence")
        # logging.warning(ls)
        return ls[2:]
    else:
        if confidence < 0:
            ls[2] = 0
        else:
            ls[2] = 1
        # returns tag,confidence,label,tweet
        return ['low_confidence', confidence, *ls[2:]]


def polarity(label):
    if label == 1:
        return 1
    else:
        return -1


class Evaluation(CoMapFunction):
    PLSTREAM_ACC = 0.75  # based on test results
    RF_CLASIFIER_ACC = 0.87  # based on test results
    # weights
    w1 = PLSTREAM_ACC / (PLSTREAM_ACC + RF_CLASIFIER_ACC)
    w2 = RF_CLASIFIER_ACC / (PLSTREAM_ACC + RF_CLASIFIER_ACC)

    def __init__(self):
        self.dict1 = defaultdict(lambda: None)  # for first stream
        self.dict2 = defaultdict(lambda: None)  # for second stream
        self.confidence = 0.5

    def calculate_confidence(self, ls, myDict, otherDict, func=default_confidence):
        """
        :param ls: list from merged_stream e.g [int(index),float(confidence),int(new_label),string(tweet),{dict}]
        from PLStream or [int(index),float(confidence),int(new_label),string(tweet),int(true_label)] from classifier
        :param myDict: dict corresponding to the current element of the string
            e.g. {int :[float(confidence),int(new_label),{dict}]}
        :param otherDict: dict corresponding to the sibling element from another substream
            e.g. {int :[float(confidence),int(new_label),int(true_label)]}

        myDict and otherDict might contain either of the two formats in the examples above
        :param func: expects a list of current element, one dictionary corresponding to the current element and another
        to the sibling element
        :return: a float value with confidence
        """

        return func(ls, myDict, otherDict)

    def map(self, ls, myDict, otherDict, log=False):
        if log:
            logging.warning("in map: " + str(ls))
            logging.warning(self.dict1.items())
            logging.warning(self.dict2.items())
        s = collect(ls, myDict, otherDict, log)
        if s == 'eval':
            confidence = self.calculate_confidence(ls, myDict, otherDict)
            return generate_label_from_confidence(confidence, ls)
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
        .map(Evaluation()).filter(lambda x: x != 'collecting' and x != 'done')
    # ds.print()
    ds = ds.filter(lambda x: x[0] != 'low_confidence')

    # .key_by(lambda x: x[0])
    return ds


def generate_new_label(ds, ds_print=None):
    ds = ds.map(lambda x: x[:-1])
    if not ds_print:
        ds = ds.map(lambda x: str(x[:-1])).add_sink(StreamingFileSink  # .set_parallelism(2)
                                                    .for_row_format('./senti_output', Encoder.simple_string_encoder())
                                                    .build())
    return ds


def calculate_accuracy(ds):
    data = ds.execute_and_collect()
    true_label = []
    predicted_label = []
    results = []
    # print(data)
    for result in data:
        true_label.append(results[-1])
        predicted_label.append(results[0])
        results.append(results)
    if true_label:
        return accuracy_score(true_label, predicted_label)
    return 'no data for accuracy'


def calculate_PLStream_accuracy(ds):
    elements = ds.execute_and_collect()
    true_label = []
    predicted_label = []
    for element in elements:
        true_label.append(element[4]['true_label'])
        predicted_label.append(element[2])
    return accuracy_score(true_label, predicted_label)


def calculate_classifier_accuracy(ds):
    elements = ds.execute_and_collect()
    true_label = []
    predicted_label = []
    for element in elements:
        true_label.append(element[4])
        predicted_label.append(element[2])
    return accuracy_score(true_label, predicted_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation in two modes, labels and accuracy. Accuracy mode is\
     default. Labels generated can be printed or stored in ./senti_output folder.')
    parser.add_argument('-l', dest='mode', action='store_const', default='ACC', const='LABEL',
                        help='Generate label(default: get accuracy, else get label)')
    parser.add_argument('-p', dest='ds_print', action='store_const', default='PRINT', const=None,
                        help='Generate label(default: print generated labels, else store to ./senti_output folder)')
    args = parser.parse_args()
    mode = args.mode
    ds_print = args.ds_print
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    start_time = time()

    # input_path = './yelp_review_polarity_csv/test.csv'
    # if input_path is not None:
    # f = pd.read_csv('./yelp_review_polarity_csv/test.csv', header=None)  # , encoding='ISO-8859-1'
    f = pd.read_csv('./exp_test.csv', header=None)  # , encoding='ISO-8859-1'
    f.columns = ["label", "review"]
    test_N = 10
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
    # print(calculate_PLStream_accuracy(ds1))
    ds2 = classifier(ds)
    # ds2.print()
    # print(calculate_classifier_accuracy(ds2))
    ds = merged_stream(ds1, ds2)
    if mode == 'LABEL':
        ds = generate_new_label(ds, ds_print)
        if ds_print:
            ds.print()
        env.execute()
    else:
        print(calculate_accuracy(ds))

    #  always specify output_type when writing to file
    logging.info("time taken for execution is: " + str(time() - start_time))
