# pylint: disable=import-error

def polarity(label):
    """
    Calculate polarity of binary label.

    Args:
        label (int): Label that is either 0 or 1.

    Returns:
        int: 1 if label is 1 else -1.
    """
    return 1 if label == 1 else -1