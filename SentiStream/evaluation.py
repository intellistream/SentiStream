import sys
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from pyflink.common.typeinfo import Types
from pyflink.common.serialization import Encoder
from pyflink.datastream import CoMapFunction, StreamExecutionEnvironment, CheckpointingMode
from pyflink.datastream.connectors import StreamingFileSink

from PLStream import unsupervised_stream
from classifier import classifier

from sklearn.metrics import accuracy_score

# logger
logger = logging.getLogger('SentiStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('sentistream.log', mode='w')
formatter = logging.Formatter('SentiStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

# supress warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect(ls, my_dict, other_dict):
    """
    Collects the data from the input stream and saves it in the respective dictionary.

    Parameters:
        ls (list): list of data
        my_dict (defaultdict): dictionary for the current stream
        other_dict (defaultdict): dictionary for the sibling stream

    Returns:
        (str): 'eval' if the data is ready for evaluation, 'done' if the data is already evaluated, 
        'collecting' if the data is still being collected.
    """
    # if sibiling dict is getting collected but not evaluated
    if other_dict[ls[0]] is not None and other_dict[ls[0]] is not False:
        # if current dict is empty, collect streams and evaluate
        if my_dict[ls[0]] is None:
            my_dict[ls[0]] = ls[1:3] + [ls[-1]]
            return 'eval'
        # if current dict is not empty then its already evaluated
        else:
            return 'done'
    # else collect streams in dicts
    else:
        if my_dict[ls[0]] is None:
            my_dict[ls[0]] = ls[1:3] + [ls[-1]]
    return 'collecting'


def generate_label_from_confidence(confidence, ls):
    """
    Generates label data based on the confidence score.

    Parameters:
        confidence (float): float value representing the confidence score
        ls (list): list of data

    Returns:
        (list): A list containing the label data, which can be a binary label or a label 
        indicating low confidence.
    """
    # assume label is positive if eval score is in range between 2 - .5
    if confidence >= 0.5:
        ls[2] = 1
        # label and review   ---- ADDED IDX FOR DEBUGGING ----- REMOVE -> [ls[0], *ls[2:4]]
        return ls[2:4]
    # assume negative if range between (-2) - (-.5)
    elif confidence <= -0.5:
        ls[2] = 0
        return ls[2:4]
    # else it has low confidence, not recommended for pseudo labeling
    else:
        if confidence < 0:
            ls[2] = 0
        else:
            ls[2] = 1
        return ['low_confidence', confidence, *ls[2:]]


def polarity(label):
    """
    Determines the polarity of a label.

    Parameters:
        label (int): integer value representing the label

    Returns:
        (int): 1 if the label is positive, -1 if the label is negative.
    """
    return 1 if label == 1 else -1


class Evaluation(CoMapFunction):
    """
    Class for evaluating the sentiment of the input stream data.
    """

    ADAPTIVE_PLSTREAM_ACC_THRESHOLD = 1
    ADAPTIVE_CLASSIFIER_ACC_THRESHOLD = 1

    def __init__(self, acc=False):
        """
        Initialize class.
        """
        self.dict1 = defaultdict(lambda: None)  # for first stream
        self.dict2 = defaultdict(lambda: None)  # for second stream
        self.acc = acc

    def calc_acc(self, ls, my_dict, other_dict):
        plstream_conf = 0
        clf_conf = 0

        true_label = None
        pred = None

        if isinstance(my_dict[ls[0]][2], dict):
            plstream_conf = my_dict[ls[0]][0] * polarity(
                my_dict[ls[0]][1]) * Evaluation.ADAPTIVE_PLSTREAM_ACC_THRESHOLD
            clf_conf = other_dict[ls[0]][0] * polarity(
                other_dict[ls[0]][1]) * Evaluation.ADAPTIVE_CLASSIFIER_ACC_THRESHOLD
            true_label = my_dict[ls[0]][2]['true_label']
            pred = my_dict[ls[0]
                           ][1] if plstream_conf > clf_conf else other_dict[ls[0]][1]

        else:
            plstream_conf = other_dict[ls[0]][0] * polarity(
                other_dict[ls[0]][1]) * Evaluation.ADAPTIVE_PLSTREAM_ACC_THRESHOLD
            clf_conf = my_dict[ls[0]][0] * polarity(
                my_dict[ls[0]][1]) * Evaluation.ADAPTIVE_CLASSIFIER_ACC_THRESHOLD
            true_label = other_dict[ls[0]][2]['true_label']
            pred = my_dict[ls[0]
                           ][1] if plstream_conf < clf_conf else other_dict[ls[0]][1]

        return ('1', accuracy_score([true_label], [pred]))

    def calculate_confidence(self, ls, my_dict, other_dict):
        """
        Calculates confidence score based on the labels and polarity of the labels.

        Parameters: 
            ls (list): list of data
            my_dict (defaultdict): dictionary for the current stream
            other_dict (defaultdict): dictionary for the sibling stream

        Returns:
            (float): A float value representing the confidence score.
        """
        # dict[idx][0] -> confidence
        # dict[idx][1] -> label

        plstream_conf = 0
        clf_conf = 0

        if isinstance(my_dict[ls[0]][2], dict):
            plstream_conf = my_dict[ls[0]][0] * polarity(
                my_dict[ls[0]][1]) * Evaluation.ADAPTIVE_PLSTREAM_ACC_THRESHOLD
            clf_conf = other_dict[ls[0]][0] * polarity(
                other_dict[ls[0]][1]) * Evaluation.ADAPTIVE_CLASSIFIER_ACC_THRESHOLD

        else:
            plstream_conf = other_dict[ls[0]][0] * polarity(
                other_dict[ls[0]][1]) * Evaluation.ADAPTIVE_PLSTREAM_ACC_THRESHOLD
            clf_conf = my_dict[ls[0]][0] * polarity(
                my_dict[ls[0]][1]) * Evaluation.ADAPTIVE_CLASSIFIER_ACC_THRESHOLD

        # MAX = 2  (1 * 1 * 1 + 1 * 1 * 1) MIN = -2 (1 * 1 * -1 + 1 * 1 * -1)
        # so range -> -2 <-> 2..
        # set low conf threashold as previous (-0.5 - 0.5)
        # return my_dict[ls[0]][0] * polarity(my_dict[ls[0]][1]) + other_dict[ls[0]][0] * polarity(other_dict[ls[0]][1])
        return plstream_conf + clf_conf

    def map(self, ls, my_dict, other_dict):
        """
        Map function to collect streaming data from two different sources and to evaluate 
        its sentiment.

        Parameters:
            ls (list): list of elements from two different streams
            my_dict (defaultdict): dictionary corresponding to the current element of the first 
            stream
            other_dict (defaultdict): dictionary corresponding to the current element of the 
            second stream

        Returns:
            (str or list): 'eval' if the data from both the streams is available, else 
            'collecting' or 'done'. If the data from both the streams is available, then a list
            with the label and sentiment score is returned.
        """

        s = collect(ls, my_dict, other_dict)
        if not self.acc:
            if s == 'eval':
                confidence = self.calculate_confidence(ls, my_dict, other_dict)
                my_dict[ls[0]] = False
                other_dict[ls[0]] = False
                return generate_label_from_confidence(confidence, ls)
            else:
                return s
        if self.acc:
            if s == 'eval':
                t = self.calc_acc(ls, my_dict, other_dict)
                my_dict[ls[0]] = False
                other_dict[ls[0]] = False
                return t
            else:
                return s

    def map1(self, ls):
        """
        Map function for the first stream.

        Parameters:
            ls (list): list of elements from the first stream

        Returns:
            (str or list): 'eval' if the data from both the streams is available, else 
            'collecting' or 'done'. If the data from both the streams is available, then a list
            with the label and sentiment score is returned.
        """
        return self.map(ls, self.dict1, self.dict2)

    def map2(self, ls):
        """
        Map function for the second stream.

        Parameters:
            ls (list): list of elements from the second stream

        Returns:
            (str or list): 'eval' if the data from both the streams is available, else 
            'collecting' or 'done'. If the data from both the streams is available, then a list
            with the label and sentiment score is returned.
        """
        return self.map(ls, self.dict2, self.dict1)


def merged_stream(ds1, ds2):
    """
    Takes two datastreams and returns new datastream that is result of merging and evaluating the 
    two input streams.

    Parameters:
        ds1 (datastream): first datastream to be merged
        ds2 (datastream): second datastream to be merged

    Returns:
        (datastream): merged and filtered datastream with high confidence
    """

    # dd = ds1.connect(ds2).map(Evaluation(acc=True)).filter(
    #     lambda x: x != 'collecting' and x != 'done') \
    #     .key_by(lambda x: x[0], key_type=Types.STRING()) \
    #     .reduce(lambda x,y: (x[0], x[1]+y[1]))

    # dd.print()

    ds = ds1.connect(ds2).map(Evaluation()).filter(
        lambda x: x != 'collecting' and x != 'done')
    ds = ds.filter(lambda x: x[0] != 'low_confidence')
    return ds


def generate_new_label(ds):
    ds.map(lambda x: f'{str(x[0])} \t {x[1]}', output_type=Types.STRING()).add_sink(StreamingFileSink  # .set_parallelism(2)
                                                                                    .for_row_format('./senti_output', Encoder.simple_string_encoder())
                                                                                    .build())
    return ds


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO, format="%(message)s")

    df = pd.read_csv('train.csv', names=['label', 'review'])

    # testing ONLY 100 reviews
    df = df.iloc[:100, :]

    df['label'] -= 1

    true_label = df.label
    yelp_review = df.review

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

    ds2 = classifier(ds)
    # ds2.print()

    ds = merged_stream(ds1, ds2)
    # ds.print()

    ds = generate_new_label(ds)

    env.execute()
