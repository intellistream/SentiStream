from torch.utils.data.dataset import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class SentimentDataset(Dataset):

    def __init__(self, labels, documents):
        super(SentimentDataset, self).__init__()

        self.documents = documents
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.documents[index], self.labels[index]
