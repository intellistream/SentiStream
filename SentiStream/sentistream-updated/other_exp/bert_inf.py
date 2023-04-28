import torch
import pandas as pd
import numpy as np
from utils import preprocess

model = torch.load('bert.pth')
model.eval()

device = torch.device("cuda")

new_df = pd.read_csv('../train.csv', names=['label', 'review'])
# df = df.iloc[5600:, :]
new_df['label'] -= 1

acc = []
with torch.no_grad():
    for i in range(0, len(new_df), 1000):
        df = new_df.iloc[i: i+1000, :]
        labels = df.label.values
        review = df.review.values

        input_ids, _ = preprocess(review)

        logits = model(torch.cat(input_ids, dim=0).to(device))[0]

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, axis=1).tolist()

        acc.append(np.sum(preds == labels) / len(preds))
        print(acc[-1])


# print(np.sum(preds == labels) / len(labels))

print(acc)

print(sum(acc) / len(acc))