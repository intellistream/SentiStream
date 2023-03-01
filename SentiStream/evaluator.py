import numpy as np
import logging
import pandas as pd
import sys

from modified_PLStream import unsupervised_stream
from dummy_classifier import dummy_classifier
# from sklearn.metrics import accuracy_score
from pyflink.datastream import CoMapFunction
from collections import defaultdict
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.connectors import StreamingFileSink

logger = logging.getLogger('SentiStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('sentistream.log', mode='w')
formatter = logging.Formatter('SentiStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def default_confidence(ls, myDict, otherDict):
    """
    Calculates confidence score based on the labels and polarity of the labels.
    
    Parameters:
        ls (list): list of data
        myDict (defaultdict): dictionary for the current stream
        otherDict (defaultdict): dictionary for the sibling stream
    
    Returns:
        (float): A float value representing the confidence score.
    """
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


def collect(ls, myDict, otherDict):
    """
    Collects the data from the input stream and saves it in the respective dictionary.
    
    Parameters:
        ls (list): list of data
        myDict (defaultdict): dictionary for the current stream
        otherDict (defaultdict): dictionary for the sibling stream
    
    Returns:
        (str): 'eval' if the data is ready for evaluation, 'done' if the data is already evaluated, 
        'collecting' if the data is still being collected.
    """
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
    """
    Generates label data based on the confidence score.
    
    Parameters:
        confidence (float): float value representing the confidence score
        ls (list): list of data
    
    Returns:
        (list): A list containing the label data, which can be a binary label or a label 
        indicating low confidence.
    """
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


def polarity(label):
    """
    Determines the polarity of a label.
    
    Parameters:
        label (int): integer value representing the label
    
    Returns:
        (int): 1 if the label is positive, -1 if the label is negative.
    """
    if label == 1:
        return 1
    else:
        return -1


class Evaluation(CoMapFunction):
    """
    Class for evaluating the sentiment of the input stream data.
    """
    PLSTREAM_ACC = 0.75  # based on test results
    DUMMY_CLASIFIER_ACC = 0.87  # based on test results
    # weights
    w1 = PLSTREAM_ACC / (PLSTREAM_ACC + DUMMY_CLASIFIER_ACC)
    w2 = DUMMY_CLASIFIER_ACC / (PLSTREAM_ACC + DUMMY_CLASIFIER_ACC)

    def __init__(self):
        """
        Initializes the class.
        """
        self.dict1 = defaultdict(lambda: None)  # for first stream
        self.dict2 = defaultdict(lambda: None)  # for second stream
        self.confidence = 0.5

    def calculate_confidence(self, ls, myDict, otherDict, func=default_confidence):
        """
        Calculates confidence score of streaming data from two different sources.
        
        Parameters:
            ls (list): list of data 
            myDict (defaultdict): dictionary for the current stream
            otherDict (defaultdict): dictionary for the sibling stream
            func (function): function for calculating confidence score
        
        Returns:
            (float): A float value representing the confidence score.
        """

        return func(ls, myDict, otherDict)

    def map(self, ls, myDict, otherDict):
        """
        Map function to collect streaming data from two different sources and to evaluate 
        its sentiment.

        Parameters:
            ls (list): list of elements from two different streams
            myDict (defaultdict): dictionary corresponding to the current element of the first 
            stream
            otherDict (defaultdict): dictionary corresponding to the current element of the 
            second stream

        Returns:
            (str or list): 'eval' if the data from both the streams is available, else 
            'collecting' or 'done'. If the data from both the streams is available, then a list
            with the label and sentiment score is returned.
        """

        s = collect(ls, myDict, otherDict)
        if s == 'eval':
            confidence = self.calculate_confidence(ls, myDict, otherDict)
            return generate_label_from_confidence(confidence, ls)
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
        logging.warning("map1")
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
        logging.warning("map2")
        return self.map(ls, self.dict2, self.dict1, True)


def merged_stream(ds1, ds2):
    """
    Takes two datastreams and returns new datastream that is result of merging and evaluating the 
    two input streams.

    Args:
        ds1 (datastream): first datastream to be merged
        ds2 (datastream): second datastream to be merged

    Returns:
        (datastream): merged and filtered datastream
    """
    ds = ds1.connect(ds2).map(Evaluation()).filter(lambda x: x != 'collecting' and x != 'done')
    ds = ds.filter(lambda x: x[0] != 'low_confidence')
    # .key_by(lambda x: x[0])
    return ds



if __name__ == '__main__':
    mode = 'LABEL'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    df = pd.read_csv('train.csv', names=['label', 'review'])

    # testing ONLY 100 reviews
    df = df.iloc[:100,:]

    df.loc[df['label'] == 1, 'label'] = 0
    df.loc[df['label'] == 2, 'label'] = 1

    true_label = df.label
    yelp_review = df.review

    data_stream = []

    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))

    print(len(yelp_review))

    print('Coming Stream is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)

    ds1 = unsupervised_stream(ds)
    # ds1.print()

    ds2 = dummy_classifier(ds)
    # ds2.print()

    ds = merged_stream(ds1, ds2)
    ds.print()

    env.execute()
