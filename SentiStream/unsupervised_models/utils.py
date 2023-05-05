# pylint: disable=import-error

import numpy as np

from numpy.linalg import norm

from Levenshtein import ratio


np.seterr(all='ignore')

txt_cache = {}
vec_cache = {}


def cos_similarity(vec1, vec2):
    """
    Compute cosine similarity between 2 same dimension vectors.

    Args:
        vec1 (ndarray): First word embedding.
        vec2 (ndarray): Ref word embedding.

    Returns:
        float: Cosine similarity between 2 vectors.
    """
    vec2_norm = vec_cache.get(tuple(vec2))

    if vec2_norm is None:
        vec2_norm = norm(vec2)
        vec_cache[tuple(vec2)] = vec2_norm

    return np.dot(vec1, vec2) / (norm(vec1) * vec2_norm)


def text_similarity(word1, word2, cutoff):
    """
    Compute text similiarity between 2 words using Levenshtein's distance.

    Args:
        word1 (str): Polar reference word.
        word2 (list): List of words in a document
        cutoff (float, optional): Threshold to consider similairty in calculation..

    Returns:
        float: Text similarity between document and reference word.
    """
    temp = []

    for word in word2:
        txt_sim = txt_cache.get((word1, word, cutoff))

        if txt_sim is None:
            txt_sim = ratio(word1, word, score_cutoff=cutoff)
            txt_cache[(word1, word, cutoff)] = txt_sim

        if txt_sim != 0:
            temp.append(txt_sim)

    return sum(temp)/len(temp) if temp else 0
