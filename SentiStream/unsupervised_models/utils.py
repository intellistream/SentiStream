# pylint: disable=import-error
from collections import defaultdict

import numpy as np

from numpy.linalg import norm
from Levenshtein import ratio

np.seterr(all='ignore')

txt_cache = defaultdict(lambda: None)
vec_cache = defaultdict(lambda: None)


def cos_similarity(vec1, vec2):
    """
    Compute cosine similarity between 2 same dimension vectors.

    Args:
        vec1 (ndarray): First word embedding.
        vec2 (ndarray): Ref word embedding.

    Returns:
        float: Cosine similarity between 2 vectors.
    """
    vec2_norm = vec_cache[tuple(vec2)]

    if vec2_norm is None:
        vec2_norm = norm(vec2)
        vec_cache[tuple(vec2)] = vec2_norm

    return np.dot(vec1, vec2) / (norm(vec1) * vec2_norm)


def text_similarity(word1, ref_words, cutoff):
    """
    Compute text similiarity between 2 words using Levenshtein's distance.

    Args:
        word1 (str): Word from document.
        word2 (list): Set of reference words
        cutoff (float, optional): Threshold to consider similairty in calculation..

    Returns:
        float: Text similarity between word and reference words.
    """
    result = txt_cache[(word1, tuple(ref_words))]

    if result is None:
        temp = []
        for word in ref_words:
            txt_sim = ratio(word1, word, score_cutoff=cutoff)

            if txt_sim != 0:
                temp.append(txt_sim)

        result = sum(temp)/len(temp) if temp else 0

        txt_cache[(word1, tuple(ref_words))] = result

    return result
