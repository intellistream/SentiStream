import torch
import pandas as pd
import numpy as np
from utils import preprocess

model = torch.load('bert.pth')
model.eval()

device = torch.device("cuda")

df = pd.read_csv('../train.csv', names=['label', 'review'])
df = df.iloc[5600:100000, :]
df['label'] -= 1

labels = df.label.values
review = df.review.values

input_ids, _ = preprocess(review)
acc = []
with torch.no_grad():
    for i in range(0, len(input_ids), 1000):
        logits = model(torch.cat(input_ids[i: i + 1000], dim=0).to(device))[0]

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, axis=1).tolist()

        acc.append(np.sum(preds == labels[i: i + 1000]) / len(preds))


# print(np.sum(preds == labels) / len(labels))

print(acc)

print(sum(acc) / len(acc))