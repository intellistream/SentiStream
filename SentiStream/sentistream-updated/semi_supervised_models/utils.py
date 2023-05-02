# pylint: disable=import-error
import re
import numpy as np
import torch

from nltk.tokenize import sent_tokenize, word_tokenize


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


def mat_mul(output, weight, bias=None):
    """
    Matrix multiplication of output and weight tensors then apply bias and tanh activation function.
    Args:
        output (Torch.tensor): output tensor to be multiplied with the weight tensor.
        weight (Torch.tensor): Weight tensor to be multiplied with the output tensor.
        bias (Torch.tensor, optional): Bias tensor to be added to the output tensor.
                                        Defaults to None.
    Returns:
        Torch.tensor: Resulted tensor.
    """
    output = torch.matmul(output, weight)
    if bias is not None:
        output += bias
    output = torch.tanh(output)
    return output.squeeze()


def element_wise_mul(output1, output2):
    """
    Element-wise multiplication of tensors
    Args:
        output1 (Torch.tensor): First output tensor.
        output2 (Torch.tensor): Second output tensor.
    Returns:
        Torch.tensor: Resulted tensor.
    """
    output = output1 * output2.unsqueeze(2)
    return output.sum(dim=0, keepdim=True)


def join_tokens(doc):
    """
    Concatenate all tokens from document list to create sentences. 

    Args:
        doc (array-like): List of documents where each represented in tokens.

    Returns:
        list: Made up sentences from concatenated tokens.
    """
    return [' '.join(tokens).strip() for tokens in doc]


def get_max_lengths(docs):
    """
    Calculate max sentence and word length from given dataset and return 80th percentile of it.

    Args:
        docs (list): Documents containing sentences.

    Returns:
        tupe(int, int): 8-th percentile of max word length and max sentence length.
    """
    words_length = []
    sents_length = []

    # Iterate through each document
    for doc in docs:
        # Tokenize sentences
        sents = sent_tokenize(doc)
        sents_length.append(len(sents))

        # Iterate through each sentence
        for sent in sents:
            # Tokenize words
            words_length.append(len(word_tokenize(sent)))

    # Calculate 80th percentile of maximum sentence and word length
    return sorted(words_length)[int(0.95 * len(words_length))], \
        sorted(sents_length)[int(0.95 * len(sents_length))]


def preprocess(docs, word_dict, max_length_word=15, max_length_sentences=10):
    """
    Encode document to embeddings for HAN model to train while padding and trimming length to make 
    all documents, sentences have same length.

    Args:
        docs (list): Documents containing sentences
        word_dict (dict): Dictionary containing words as keys and their corresponding 
                                    indices as values.
        max_length_word (int, optional): Maximum length of word tokens. Defaults to 15.
        max_length_sentences (int, optional): Maximu length of sentence tokens. Defaults to 10.

    Returns:
        ndarray: Word embeddings in document format.
    """
    # Create padded sequence to reuse it for padding words or sentences.
    padded_words = [-1] * max_length_word

    temp = []

    # Iterate over each document.
    for doc in docs:
        # UNK = -1 , PAD = -1  ### HAVE SEPARATE ENCODINGSSS

        # Tokenize document into sentences and encode each sentence into list of word indices.
        document_encode = [
            [word_dict.get(word, -1)
             for word in re.split(r'[.!?\s]+', sentences)
             if len(word) > 1] for sentences
            in
            sent_tokenize(doc)]

        # Pad short sentences with -1s to make them same length as max_length_word.
        for sentence in document_encode:
            if len(sentence) < max_length_word:
                sentence += padded_words[len(sentence):]

        # Pad short documents with -1s to make them same length as max_length_sentences.
        if len(document_encode) < max_length_sentences:
            document_encode += [padded_words] * \
                (max_length_sentences - len(document_encode))

        # Trim sentences and documents to be same as max_length_word and max_length_sentences.
        document_encode = [sentences[:max_length_word] for sentences in document_encode][
            :max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)

        # Increment all elements of matrix by 1 to make all values positive
        document_encode += 1

        temp.append(document_encode)

    return temp


def downsampling(label, text):
    """
    Downsample majority class in binary classification to balance class.

    Args:
        label (list): List of labels.
        text (list): List of documents.

    Returns:
        tuple: Downsampled labels and documents.
    """
    pos_idx = [idx for idx, x in enumerate(label) if x == 1]
    neg_idx = [idx for idx, x in enumerate(label) if x == 0]

    # no need to shuflle since it will be shuffled in train_test_split.
    if len(pos_idx) < len(neg_idx):
        downsampled_idx = pos_idx + neg_idx[:len(pos_idx)]
    else:
        downsampled_idx = neg_idx + pos_idx[:len(neg_idx)]

    return [label[i] for i in downsampled_idx], [text[i] for i in downsampled_idx]
