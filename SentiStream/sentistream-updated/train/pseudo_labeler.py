# pylint: disable=import-error
# pylint: disable=no-name-in-module

from collections import defaultdict
from pyflink.datastream.functions import CoMapFunction

import config

from train.utils import polarity, calculate_acc


class SentimentPseudoLabeler:
    """
    Generate pseudo labels for unlabeled data using unsupervised and semi-supervised models.

    Attributes:
        to_calc_acc (list): Store predicted and ground truth labels to calculate accuracy 
                            batch-wise.
        collection (dict): Dictionary containing both predictions and confidence score for
                            same text.

    Constants:
        ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT: Dynamic weight for unsupervised model for 
                                                ensembled predictions.
        ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT: Dynamic weight for semi-supervised model for 
                                                ensembled predictions.
    """
    ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT = 1
    ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT = 5

    def __init__(self):
        """
        Initialize class to generate pseudo labels.
        """
        self.to_calc_acc = []
        self.collector = defaultdict(dict)

    def get_confidence_score(self, data):
        """
        Calculate confidence score for final prediction from both learning methods.

        Args:
            data (dict): Contains predicted results of both models.
                            - {'us': [us_conf, us_pred, label], 'ss': [ss_conf, ss_pred, text]}

        Returns:
            float: Confidence score of final prediction.
        """

        # Calculate unsupervised model's weighted confidence.
        us_conf = data['us'][0] * polarity(data['us'][1]) * \
            SentimentPseudoLabeler.ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT

        # Calculate semi-supervised model's weighted confidence.
        ss_conf = data['ss'][0] * polarity(data['ss'][1]) * \
            SentimentPseudoLabeler.ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT

        # Store final prediction to calculate sentistream's accuracy.
        self.to_calc_acc.append([
            [data['us'][2]], [data['us'][1] if us_conf > ss_conf else data['ss'][1]]])

        return us_conf + ss_conf

    def get_model_acc(self):
        """
        Calculate model's final predictions' accuracy.

        Returns:
            float: Accuracy of final output.
        """
        if self.to_calc_acc:
            acc = calculate_acc(self.to_calc_acc)
            self.to_calc_acc = []
            return acc
        return None

    def generate_pseudo_label(self, first_output, second_output):
        """
        Generate pseudo label for incoming output from both models.

        Args:
            first_output/second_output (tuple): contains data from unsupervised model's output.
                                - us_idx: index of outputs from unsupervised model.
                                - us_flag: indicates unsupervised model's output.
                                - us_conf: unsupervised model's confidence for predicted label.
                                - us_pred: unsupervised model's prediction.
                                - label: ground truth label.
            first_output/second_output (tuple): contains data from semi-supervised model's output.
                                - ss_idx: index of outputs from semi-supervised model.
                                - ss_flag: indicates semi-supervised model's output.
                                - ss_conf: semi-supervised model's confidence for predicted label.
                                - ss_pred: semi-supervised model's prediction.
                                - text: text data / review.

        Returns:
            list: pseudo label for current data.
        """

        # Store outputs in dictionary to map them easily.
        for stream_output in (first_output, second_output):
            if stream_output:
                self.collector[stream_output[0]
                               ][stream_output[1]] = stream_output[2:]

        output = []

        # Generate labels for data that have both predicted results.
        for stream_output in (first_output, second_output):
            if stream_output and len(self.collector[stream_output[0]]) == 2:
                conf = self.get_confidence_score(
                    self.collector[stream_output[0]])
                output.append(self.get_pseudo_label(conf, stream_output[0]))

        if not output:
            return [config.BATCHING]
        return output

    def get_pseudo_label(self, conf, key):
        """
        Generate pseudo label based on finalized confidence score.

        Args:
            conf (float): Model's finalized confidence score.
            key (int): Key for dictionary of predicted outputs.

        Returns:
            str or list: 'LOW_CONFIDENCE' if confidence score is too low to make it as pseudo label,
                        else, model's predicted senitment along with text.
        """

        text = self.collector[key]['ss'][2]

        # Delete item from collector to avoid re-generating labels.
        del self.collector[key]

        return config.LOW_CONF if -0.5 < conf < 0.5 else [key, 1 if conf >= 0.5 else 0, text]


class PseudoLabelerCoMap(CoMapFunction):
    """
    CoMapFunction that uses labeler function to generate pseudo labels for stream data.
    """

    def __init__(self, labeler):
        """
        Initialize class with labeler function

        Args:
            labeler (SentimentPseudoLabeler): Instance of SentimentPseudoLabeler class to 
                                            generate labels.
        """
        self.labeler = labeler

    def map1(self, value):
        """
        Generate pseudo label to on of merged datastream.

        Args:
            value (tuple): Contain's pretrained model's output.

        Returns:
            list: pseudo label and text.
        """
        return self.labeler.generate_pseudo_label(value, None)

    def map2(self, value):
        """
        Generate pseudo label to on of merged datastream.

        Args:
            value (tuple): Contain's pretrained model's output.

        Returns:
            list: pseudo label and text.
        """
        return self.labeler.generate_pseudo_label(value, None)
