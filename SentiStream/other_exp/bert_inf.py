# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch
import pandas as pd
import numpy as np

from torch.nn.parallel import DataParallel
from time import time

from utils import preprocess

model = torch.load('bert_1.pth')
model = DataParallel(model)
model.eval()

device = torch.device("cuda")

new_df = pd.read_csv('../new_train_1_percent.csv', names=['label', 'review'])

acc = []

start = time()
with torch.no_grad():
    for i in range(0, len(new_df), 2000):
        df = new_df.iloc[i: i+2000, :]
        labels = df.label.values
        review = df.review.values

        input_ids, _ = preprocess(review)

        logits = model(torch.cat(input_ids, dim=0).to(device))[0]

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, axis=1).tolist()

        acc.append(np.sum(preds == labels) / len(preds))
        print(acc[-1])
print('ELAPSED TIME: ', time() - start)

print(acc)

print(sum(acc) / len(acc))


# BERT 1%

# [0.8525, 0.8425, 0.8425, 0.8375, 0.8375, 0.836, 0.8375, 0.851, 0.837, 0.832, 0.8445, 0.831, 0.848, 0.831, 0.836, 0.8455, 0.84, 0.851, 0.833, 0.8425, 0.8375, 0.834, 0.8375, 0.8505, 0.8445, 0.85, 0.848, 0.8415, 0.857, 0.842, 0.8615, 0.838, 0.841, 0.841, 0.844, 0.8355, 0.8415, 0.8345, 0.852, 0.815, 0.8205, 0.819, 0.8195, 0.8225, 0.815, 0.817, 0.8105, 0.8165, 0.8095, 0.8045,
#     0.8085, 0.798, 0.8345, 0.823, 0.8105, 0.815, 0.8045, 0.809, 0.811, 0.7965, 0.8235, 0.8275, 0.8095, 0.76, 0.537, 0.5295, 0.521, 0.518, 0.537, 0.5065, 0.5205, 0.514, 0.52, 0.5255, 0.5155, 0.529, 0.523, 0.5125, 0.515, 0.5215, 0.5205, 0.504, 0.524, 0.5215, 0.519, 0.5175, 0.5275, 0.5075, 0.5325, 0.511, 0.519, 0.5225, 0.5265, 0.5425, 0.5175, 0.5355, 0.5125, 0.5201640464798359]
