# pylint: disable=import-error

import numpy as np
from numpy.linalg import norm

from Levenshtein import ratio


np.seterr(all='ignore')


def cos_similarity(vec1, vec2):
    """
    Compute cosine similarity between 2 same dimension vectors.

    Args:
        vec1 (ndarray): First word embedding.
        vec2 (ndarray): Second word embedding.

    Returns:
        float: Cosine similarity between 2 vectors.
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def text_similarity(word1, word2, cutoff=0.9):
    """
    Compute text similiarity between 2 words using Levenshtein's distance.

    Args:
        word1 (str): Polar reference word.
        word2 (list): List of words in a document
        cutoff (float, optional): Threshold to consider similairty in calculation. Defaults to 0.9.

    Returns:
        float: Text similarity between document and reference word.
    """
    temp = []

    for word in word2:
        txt_sim = ratio(word1, word, score_cutoff=cutoff)

        if txt_sim != 0:
            temp.append(txt_sim)

    return sum(temp)/len(temp) if temp else 0
