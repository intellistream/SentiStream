# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch
import pandas as pd
import numpy as np

from torch.nn.parallel import DataParallel

from utils import preprocess

model = torch.load('bert.pth')
model = DataParallel(model)
model.eval()

device = torch.device("cuda")

new_df = pd.read_csv('../new_train.csv', names=['label', 'review'])

acc = []
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

print(acc)

print(sum(acc) / len(acc))
