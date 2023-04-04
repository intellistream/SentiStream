from torch.utils.data.dataset import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class SentimentDataset(Dataset):

    def __init__(self, labels, documents, wb_dict, max_length_sentences=30, max_length_word=35):
        super(SentimentDataset, self).__init__()

        self.documents = documents.tolist()
        self.labels = labels.tolist()
        self.dict = wb_dict

        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word

        self.padded_words = [-1] * self.max_length_word

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        document = self.documents[index]

        # UNK = -1 , PAD = -1  ### HAVE SEPARATE ENCODINGSSS

        document_encode = [
            [np.where(self.dict == word)[0][0] if word in self.dict else -1 for word in word_tokenize(sentences)] for sentences
            in
            sent_tokenize(document)]

        for sentence in document_encode:
            if len(sentence) < self.max_length_word:
                sentence += self.padded_words[len(sentence):]

        if len(document_encode) < self.max_length_sentences:
            document_encode += [self.padded_words] * \
                (self.max_length_sentences - len(document_encode))

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
            :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1  # make all pos

        return document_encode.astype(np.int64), label
