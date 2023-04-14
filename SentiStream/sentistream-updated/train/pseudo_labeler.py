# pylint: disable=import-error
# pylint: disable=no-name-in-module

from train.utils import polarity, calculate_acc


class SentimentPseudoLabeler:
    """
    Generate pseudo labels for unlabeled data using unsupervised and semi-supervised models.

    Attributes:
        to_calc_acc (list): Store predicted and ground truth labels to calculate accuracy batch-wise.

    Constants:
        ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT: Dynamic weight for unsupervised model for 
                                                ensembled predictions.
        ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT: Dynamic weight for semi-supervised model for 
                                                ensembled predictions.
    """
    ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT = 0.5
    ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT = 0.5

    def __init__(self):
        """
        Initialize class to generate pseudo labels.
        """
        self.to_calc_acc = []

    def get_confidence_score(self, data):
        """
        Calculate confidence score for final prediction from both learning methods.

        Args:
            data (tuple): Contains data from both model's outputs.
                            - us_conf: unsupervised model's confidence for predicted label.
                            - us_pred: unsupervised model's prediction.
                            - ss_conf: semi-supervised model's confidence for predicted label.
                            - ss_pred: semi-supervised model's prediction.
                            - text: text data / review.
                            - label: ground truth label.

        Returns:
            float: Confidence score of final prediction.
        """

        # Calculate unsupervised model's weighted confidence.
        us_conf = data[0] * polarity(data[1]) * \
            SentimentPseudoLabeler.ADAPTIVE_UNSUPERVISED_PREDICTION_WEIGHT

        # Calculate semi-supervised model's weighted confidence.
        ss_conf = data[2] * polarity(data[3]) * \
            SentimentPseudoLabeler.ADAPTIVE_SEMI_SUPERVISED_PREDICTION_WEIGHT

        # Store final prediction to calculate sentistream's accuracy.
        self.to_calc_acc.append([
            [data[4]], [data[1] if us_conf > ss_conf else data[3]]])

        return us_conf + ss_conf

    def get_model_acc(self):
        """
        Calculate model's final predictions' accuracy.

        Returns:
            float: Accuracy of final output.
        """
        acc = calculate_acc(self.to_calc_acc)
        self.to_calc_acc = []

        return acc

    def generate_pseudo_label(self, us_output, ss_output):
        """
        Generate pseudo label for incoming output from both models.

        Args:
            us_output (tuple): contains data from unsupervised model's output.
                                - us_conf: unsupervised model's confidence for predicted label.
                                - us_pred: unsupervised model's prediction.
                                - text: text data / review.
            ss_output (tuple): contains data from semi-supervised model's output.
                                - ss_conf: semi-supervised model's confidence for predicted label.
                                - ss_pred: semi-supervised model's prediction.
                                - label: ground truth label.

        Returns:
            list: pseudo label for current data.
        """

        conf = self.get_confidence_score(
            (us_output[0], us_output[1], ss_output[0], ss_output[1], ss_output[2], us_output[2]))

        return self.get_pseudo_label(conf, us_output[2])

    def get_pseudo_label(self, conf, text):
        """
        Generate pseudo label based on finalized confidence score.

        Args:
            conf (float): Model's finalized confidence score.
            text (str): Text data / Review.

        Returns:
            str or list: 'LOW_CONFIDENCE' if confidence score is too low to make it as pseudo label,
                        else, model's predicted senitment along with text.
        """
        return 'LOW_CONFIDENCE' if -0.5 < conf < 0.5 else [1 if conf >= 0.5 else 0, text]


class PseudoLabelerCoMap(CoMapFunction):