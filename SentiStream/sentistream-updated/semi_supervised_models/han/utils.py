# pylint: disable=import-error
import re
import numpy as np
import torch

from nltk.tokenize import sent_tokenize, word_tokenize


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


def calc_acc(y_label, y_pred):
    """
    Calulate accuracy of model's predictions.

    Args:
        y_label (Torch.tensor): True labels for each sample.
        y_pred (Torch.tensor): Predicted labels for each sample.

    Returns:
        Torch.tenosr: Accuracy of model's prediction.
    """
    preds = y_pred.argmax(dim=1)
    correct = (preds == y_label).sum()
    return correct / len(y_label)


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
    return sorted(words_length)[int(0.8 * len(words_length))], \
        sorted(sents_length)[int(0.8 * len(sents_length))]


def preprocess(docs, word_dict, max_length_word=15, max_length_sentences=15):
    """
    Encode document to embeddings for HAN model to train while padding and trimming length to make 
    all documents, sentences have same length.

    Args:
        docs (list): Documents containing sentences
        word_dict (dict): Dictionary containing words as keys and their corresponding 
                                    indices as values.
        max_length_word (int, optional): Maximum length of word tokens. Defaults to 15.
        max_length_sentences (int, optional): Maximu length of sentence tokens. Defaults to 15.

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
