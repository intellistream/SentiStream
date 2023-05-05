# pylint: disable=import-error
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    PyTorch Dataset class for sentiment analysis, containing precomputed word vectors and binary 
    sentiment labels.
    """

    def __init__(self, vectors, labels):
        """
        Initialize class with vectors and labels.

        Args:
            vectors (array-like): Precomputed word embeddings, where each element corresponds to
                                the embedding for single input sentence.
            labels (array-like): Corresponding binary labels (0-negative, 1-positive).
        """
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        """
        Get length of dataset

        Returns:
            length of dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get data pair at given index.

        Args:
            idx (int): Index of data pair

        Returns:
            tupe(array-like, int): vector and corresponding label
        """
        return self.vectors[idx], self.labels[idx]
