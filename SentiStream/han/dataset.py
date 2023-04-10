from torch.utils.data import Dataset


class SentimentDataset(Dataset):

    def __init__(self, labels, texts):
        super().__init__()

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
