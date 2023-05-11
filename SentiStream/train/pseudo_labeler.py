# pylint: disable=import-error
# pylint: disable=no-name-in-module
from collections import defaultdict
from itertools import zip_longest
from pyflink.datastream.functions import CoMapFunction

from train.utils import polarity, calculate_perf_metrics


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
        ADAPTIVE_POS_LE_GAP: Upper threshold for positive labeled pseduo label's confidence.
        ADAPTIVE_NEG_LE_GAP: Upper threshold for negative labeled pseduo label's confidence.
        FIXED_POS_THRESHOLD: Lower threshold for positive labeled pseduo label's confidence.
        FIXED_NEG_THRESHOLD: Lower threshold for negative labeled pseduo label's confidence.
    """
    ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT = 1
    ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT = 0.5

    ADAPTIVE_POS_LE_GAP = 0.05
    ADAPTIVE_NEG_LE_GAP = 0.05

    FIXED_POS_THRESHOLD = 0.8
    FIXED_NEG_THRESHOLD = 0.8

    def __init__(self):
        """
        Initialize class to generate pseudo labels.
        """
        self.to_calc_acc = []
        self.acc_list = []
        self.f1_list = []
        self.collector = defaultdict(dict)

        self.us = 0
        self.ss = 0

    def get_confidence_score(self, data):
        """
        Calculate confidence score for final prediction from both learning methods.

        Args:
            data (dict): Contains predicted results of both models.
                            - {'us': [us_conf, us_pred, label], 'ss': [ss_conf, ss_pred, text]}

        Returns:
            float: Confidence score of final prediction.
        """
        ss = data['ss']
        us = data['us']

        # Calculate models' confidence.
        us_conf, ss_conf = us[0] * polarity(us[1]), ss[0] * polarity(ss[1])

        us_conf *= SentimentPseudoLabeler.ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT
        ss_conf *= SentimentPseudoLabeler.ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT

        if ss[0] > 0.5 and us[0] > 0.5:
            conf = us_conf * 0.75 + ss_conf * 0.75
            pred = 1 if conf > 0 else 0

            self.us += pred == us[1]
            self.ss += pred == ss[1]

        else:
            pred = us[1] if abs(us_conf) > abs(ss_conf) else ss[1]
            conf = us_conf + ss_conf

        # Store final prediction to calculate sentistream's accuracy.
        self.to_calc_acc.append([[us[2]], [pred]])

        return conf

    def generate_pseudo_label(self, first_stream, second_stream):
        """
        Generate pseudo label for incoming output from both models.

        Args:
            first_stream/second_stream (list): contains list of tuple from unsupervised model's 
                                                output.
                                - us_idx: index of outputs from unsupervised model.
                                - us_flag: indicates unsupervised model's output.
                                - us_conf: unsupervised model's confidence for predicted label.
                                - us_pred: unsupervised model's prediction.
                                - label: ground truth label.
            first_stream/second_stream (list): contains list of tuple from semi-supervised model's 
                                                output.
                                - ss_idx: index of outputs from semi-supervised model.
                                - ss_flag: indicates semi-supervised model's output.
                                - ss_conf: semi-supervised model's confidence for predicted label.
                                - ss_pred: semi-supervised model's prediction.
                                - text: text data / review.

        Returns:
            list: pseudo labels.
        """
        conf_list = []
        key_list = []

        for first_output, second_output in zip_longest(first_stream, second_stream):
            # Store outputs in dictionary to map them easily.
            for stream_output in (first_output, second_output):
                if stream_output:
                    self.collector[stream_output[0]
                                   ][stream_output[1]] = stream_output[2:]

            # Generate labels for data that have both predicted results.
            for stream_output in (first_output, second_output):
                if stream_output and len(self.collector[stream_output[0]]) == 2:
                    if stream_output[0] not in key_list:
                        conf = self.get_confidence_score(
                            self.collector[stream_output[0]])

                        conf_list.append(conf)
                        key_list.append(stream_output[0])

        output = self.get_pseudo_label(conf_list, key_list)

        if self.us > self.ss:
            self.us *= 1.5
        else:
            self.ss *= 1.5

        max_us_ss = max(self.us, self.ss)

        if max_us_ss:
            SentimentPseudoLabeler.ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT = self.us / max_us_ss
            SentimentPseudoLabeler.ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT = self.ss / max_us_ss

        if self.to_calc_acc:
            acc, f1 = calculate_perf_metrics(self.to_calc_acc)
            self.acc_list.append(acc)
            self.f1_list.append(f1)

        self.us, self.ss, self.to_calc_acc = 0, 0, []

        return output

    def get_pseudo_label(self, conf_list, key_list):
        """
        Generate pseudo label based on finalized confidence score.

        Args:
            conf_list (list): List of model's finalized confidence score.
            key_list (list): List of keys for dictionary of predicted outputs.

        Returns:
            list: list of model's predicted senitment along with text ffrom high 
                confident predictions.
        """
        pos_ = sum(conf > SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
                   SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP for conf in conf_list)
        neg_ = sum(conf < -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
                   SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP) for conf in conf_list)

        normalized_denom = max(len(conf_list) - (pos_ + neg_), pos_, neg_)

        pos_learn_eff = pos_ / normalized_denom
        pos_learn_eff /= (2 - pos_learn_eff)

        neg_learn_eff = neg_ / normalized_denom
        neg_learn_eff /= (2 - neg_learn_eff)

        pseudo_labels = [[1 if conf >= 0 else 0,
                          self.collector[key_list[idx]]['ss'][2]]
                         for idx, conf in enumerate(conf_list)
                         if conf <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
                                      SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
                                      neg_learn_eff)
                         or conf >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
                                     SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
                                     pos_learn_eff)
                         ]

        for key in key_list:
            del self.collector[key]

        return pseudo_labels
