# pylint: disable=import-error
import torch


def calc_acc(y_pred, y_test):
    """
    Calculate accuracy of predictions.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_test (torch.Tensor): True labels.

    Returns:
        float: Accuracy of binary predictions.
    """
    y_pred_rounded = torch.round(y_pred)
    correct_results = (y_pred_rounded == y_test).sum()
    return correct_results / y_pred.size(0)
