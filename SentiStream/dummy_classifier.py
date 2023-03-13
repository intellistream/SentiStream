import random

from pyflink.datastream.functions import MapFunction


class DummyClassifier(MapFunction):
    """
    Class for Classifier Placeholder.
    """

    def __init__(self):
        """
        Initializes class with model.
        """
        self.model = None
        self.data = []

    def get_confidence(self):
        """
        Predict label and calculate confidence.

        Returns:
            (list): list of float value representing the confidence scores for polarity.
        """
        neg_conf = random.random()
        return [neg_conf, 1-neg_conf]

    def map(self, data):
        """
        Map function predict polarity of datastream.

        Parameters:
            data (list): list of elements from datastreams

        Returns:
            (list): list of predicted outputs in format of [index, prediction confidence, 
            predicted label, review, true label]
        """
        conf = self.get_confidence()
        pred_conf = max(conf)

        return [data[0], pred_conf, conf.index(pred_conf), data[2], data[1]]


def dummy_classifier(ds):
    """
    Placeholder for Sentiment classifier

    Args:
        ds (datastream): stream of data from source

    Returns:
        datastream: predicted datastream
    """
    ds = ds.map(DummyClassifier())
    return ds
