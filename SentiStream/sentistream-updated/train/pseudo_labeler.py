# pylint: disable=import-error
# pylint: disable=no-name-in-module

from collections import defaultdict
from itertools import zip_longest
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
    ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT = 0.5
    ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT = 1

    POS_LEARNING_EFFECT = 0
    NEG_LEARNING_EFFECT = 0

    ADAPTIVE_POS_LE_GAP = 0.05  # 0.05   # 0.03
    ADAPTIVE_NEG_LE_GAP = 0.05  # 0.05   # 0.02

    FIXED_POS_THRESHOLD = 0.80  # 0.90   # 0.77
    FIXED_NEG_THRESHOLD = 0.80  # 0.90   # 0.78

    def __init__(self):
        """
        Initialize class to generate pseudo labels.
        """
        self.to_calc_acc = []
        self.collector = defaultdict(dict)

        # TODO: DELETE -- TO DEBUG
        self.us_ss_same_crct = 0
        self.us_ss_same_wrng = 0
        self.crct_aft = 0
        self.wrng_aft = 0
        self.us_crct = 0
        self.ss_crct = 0
        self.ttl_true_pos = 0
        self.ttl_true_neg = 0
        self.ttl_us_pos = 0
        self.ttl_us_neg = 0
        self.ttl_ss_pos = 0
        self.ttl_ss_neg = 0
        self.pseudo_pos_crct = 0
        self.pseudo_pos_wrng = 0
        self.pseudo_neg_crct = 0
        self.pseudo_neg_wrng = 0
        self.both_same_wrong_conf = []
        self.both_same_crct_conf = []
        self.us_ss_same_crct_aft = 0
        self.us_ss_same_wrng_aft = 0
        self.pseudo_pos_crct_aft = 0
        self.pseudo_pos_wrng_aft = 0
        self.pseudo_neg_crct_aft = 0
        self.pseudo_neg_wrng_aft = 0
        self.us_crct_aft = 0
        self.ss_crct_aft = 0
        self.crt = 0
        self.wrng = 0
        self.pos = 0
        self.neg = 0

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

        # Calculate unsupervised model's weighted confidence.
        us_conf = us[0] * polarity(us[1]) * \
            SentimentPseudoLabeler.ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT

        # Calculate semi-supervised model's weighted confidence.
        ss_conf = ss[0] * polarity(ss[1]) * \
            SentimentPseudoLabeler.ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT

        if ss[1] == us[1] and ss[0] > 0.5 and us[0] > 0.5:
            pred = ss[1]
            conf = us[0] * polarity(us[1]) * 0.5 + \
                ss[0] * polarity(ss[1]) * 0.5

        else:
            pred = us[1] if abs(us_conf) > abs(
                ss_conf) else ss[1]
            conf = us_conf + ss_conf

        # Store final prediction to calculate sentistream's accuracy.
        self.to_calc_acc.append([[us[2]], [pred]])

        return conf

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

        output = []

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

        if not output:
            return [config.BATCHING]
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

        if pos_ + neg_ > 0:
            normalize_denom = max(len(conf_list) - (pos_ + neg_), pos_, neg_)

            SentimentPseudoLabeler.POS_LEARNING_EFFECT = pos_ / normalize_denom
            SentimentPseudoLabeler.POS_LEARNING_EFFECT /= (
                2 - SentimentPseudoLabeler.POS_LEARNING_EFFECT)

            SentimentPseudoLabeler.NEG_LEARNING_EFFECT = neg_ / normalize_denom
            SentimentPseudoLabeler.NEG_LEARNING_EFFECT /= (
                2 - SentimentPseudoLabeler.NEG_LEARNING_EFFECT)

            # print('\POS & NEG LABELS BEFORE FLEXMATCH: ', pos_, neg_)
            # print('FLEXMATCH LEARNING EFFECT FOR POS & NEG: ',
            #       SentimentPseudoLabeler.POS_LEARNING_EFFECT,
            #       SentimentPseudoLabeler.NEG_LEARNING_EFFECT)

            # pos_ = 0
            # neg_ = 0

            # for c in conf_list:
            #     if c <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
            #               SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
            #               SentimentPseudoLabeler.NEG_LEARNING_EFFECT) or \
            #             c >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
            #                   SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
            #                   SentimentPseudoLabeler.POS_LEARNING_EFFECT):
            #         if c < 0:
            #             neg_ += 1
            #         else:
            #             pos_ += 1

        # #     # print('POS & NEG LABELS AFTER FLEXMATCH ', pos_, neg_)

        # for idx, key in enumerate(key_list):
        #     temp = self.collector[key]
        #     ss = temp['ss'][1]
        #     us = temp['us'][1]
        #     t = temp['us'][2]

        #     if conf_list[idx] <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
        #                            SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
        #                            SentimentPseudoLabeler.NEG_LEARNING_EFFECT) or \
        #             conf_list[idx] >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
        #                                SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
        #                                SentimentPseudoLabeler.POS_LEARNING_EFFECT):
        #         pass
        #     elif -0.1 < conf_list[idx] < 0.1:
        #         if ss != t:
        #             self.crt += 1
        #         else:
        #             self.wrng += 1
        #     # elif -0.1 < conf_list[idx] < 0.1 :
        #         # if conf_list[idx] < 0 and t == 0 or conf_list[idx] > 0 and t == 1:
        #         #     self.crt += 1
        #         # else:
        #         #     self.wrng += 1

        # for idx, key in enumerate(key_list):
        #     temp = self.collector[key]
        #     ss = temp['ss'][1]
        #     us = temp['us'][1]
        #     t = temp['us'][2]

        #     if t == 1:
        #         self.ttl_true_pos += 1
        #     else:
        #         self.ttl_true_neg += 1

        #     if ss == 1:
        #         self.ttl_ss_pos += 1
        #     else:
        #         self.ttl_ss_neg += 1

        #     if us == 1:
        #         self.ttl_us_pos += 1
        #     else:
        #         self.ttl_us_neg += 1

        #     if conf_list[idx] <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
        #                            SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
        #                            SentimentPseudoLabeler.NEG_LEARNING_EFFECT) or \
        #             conf_list[idx] >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
        #                                SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
        #                                SentimentPseudoLabeler.POS_LEARNING_EFFECT):
        #         if ss == us:
        #             if ss == t:
        #                 self.us_ss_same_crct += 1
        #                 self.both_same_crct_conf.append(conf_list[idx])

        #                 if ss == 0:
        #                     self.pseudo_neg_crct += 1
        #                 else:
        #                     self.pseudo_pos_crct += 1

        #             else:
        #                 self.us_ss_same_wrng += 1
        #                 self.both_same_wrong_conf.append(conf_list[idx])

        #                 if ss == 0:
        #                     self.pseudo_neg_wrng += 1
        #                 else:
        #                     self.pseudo_pos_wrng += 1
        #         else:
        #             if ss == t:
        #                 self.ss_crct += 1
        #             else:
        #                 self.us_crct += 1

        # for idx, key in enumerate(key_list):
        #     temp = self.collector[key]
        #     ss = temp['ss'][1]
        #     us = temp['us'][1]
        #     t = temp['us'][2]

        #     if conf_list[idx] <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
        #                            SentimentPseudoLabeler.NEG_LEARNING_EFFECT) or \
        #             conf_list[idx] >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
        #                                SentimentPseudoLabeler.POS_LEARNING_EFFECT):
        #         if ss == us:
        #             if ss == t:
        #                 self.us_ss_same_crct_aft += 1

        #                 if ss == 0:
        #                     self.pseudo_neg_crct_aft += 1
        #                 else:
        #                     self.pseudo_pos_crct_aft += 1

        #             else:
        #                 self.us_ss_same_wrng_aft += 1

        #                 if ss == 0:
        #                     self.pseudo_neg_wrng_aft += 1
        #                 else:
        #                     self.pseudo_pos_wrng_aft += 1
        #         else:
        #             if ss == t:
        #                 self.ss_crct_aft += 1
        #             else:
        #                 self.us_crct_aft += 1

        pseudo_labels = [[key_list[idx], 1 if conf >= 0 else 0,
                          self.collector[key_list[idx]]['ss'][2]]
                         for idx, conf in enumerate(conf_list)
                         if conf <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
                                      SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
                                      SentimentPseudoLabeler.NEG_LEARNING_EFFECT)
                         or conf >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
                                     SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
                                     SentimentPseudoLabeler.POS_LEARNING_EFFECT)
                         ]

        # WITHOUT FLEXMATCH
        # pseudo_labels = [[key_list[idx], 1 if conf >= 0 else 0,
        #                   self.collector[key_list[idx]]['ss'][2]]
        #                  for idx, conf in enumerate(conf_list)
        #                  if conf <= -(SentimentPseudoLabeler.FIXED_NEG_THRESHOLD +
        #                               SentimentPseudoLabeler.ADAPTIVE_NEG_LE_GAP *
        #                               1)
        #                  or conf >= (SentimentPseudoLabeler.FIXED_POS_THRESHOLD +
        #                              SentimentPseudoLabeler.ADAPTIVE_POS_LE_GAP *
        #                              1)
        #                  ]

        for key in key_list:
            del self.collector[key]

        return pseudo_labels


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
