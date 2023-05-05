# pylint: disable=import-error

from sklearn.metrics import accuracy_score


def polarity(label):
    """
    Calculate polarity of binary label.

    Args:
        label (int): Label that is either 0 or 1.

    Returns:
        int: 1 if label is 1 else -1.
    """
    return 1 if label == 1 else -1


def calculate_acc(data):
    """
    Calculate accuracy score of model.

    Args:
        data (list): List of ground truth and predicted labels.

    Returns:
        float: Accuracy score of model's predictions.
    """
    y_true, y_pred = zip(*data)
    return accuracy_score(y_true, y_pred)
