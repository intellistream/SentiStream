# pylint: disable=import-error

from numpy import dot
from numpy.linalg import norm


def cos_similarity(vec1, vec2):
    """
    Compute cosine similarity between 2 same dimension vectors.

    Args:
        vec1 (ndarray): First word embedding.
        vec2 (ndarray): Second word embedding.

    Returns:
        float: Cosine similarity between 2 vectors.
    """
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
