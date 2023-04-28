from utils import get_max_len, preprocess, acc

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

import torch
import pandas as pd

df = pd.read_csv('../train.csv', names=['label', 'review'])
df = df.iloc[:5600, :]
df['label'] -= 1
# df = pd.read_csv('../new_ss_train.csv', names=['label', 'review'])

labels = df.label.values
review = df.review.values

BATCH_SIZE = 64
EPOCHS = 20
device = torch.device("cuda")


# max_len = get_max_len(review) # just checked if max len is less than 512.. but its large so truncated to 512.. so no need to run this always.
input_ids, attention_masks = preprocess(review)

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)
train_df, test_df = train_test_split(
    dataset, test_size=0.2, random_state=42, shuffle=True)

train_dataloader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_df, batch_size=BATCH_SIZE, shuffle=False)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, output_attentions=False, output_hidden_states=False)
model.cuda()


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)

best_loss = 1e5
train_loss = [0] * EPOCHS
train_acc = [0] * EPOCHS
val_loss = [0] * EPOCHS
val_acc = [0] * EPOCHS

for epoch in range(0, EPOCHS):
    model.train()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)

        model.zero_grad()

        loss, logits = model(input_ids,
                             token_type_ids=None,
                             attention_mask=input_mask,
                             labels=labels,
                             return_dict=False)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        train_loss[epoch] += loss.item()
        train_acc[epoch] += acc(logits, labels)

        optimizer.step()
        scheduler.step()

    train_loss[epoch] /= len(train_dataloader)
    train_acc[epoch] /= len(train_dataloader)

    model.eval()

    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            (loss, logits) = model(input_ids,
                                   token_type_ids=None,
                                   attention_mask=input_mask,
                                   labels=labels,
                                   return_dict=False)

        val_loss[epoch] += loss.item()
        val_acc[epoch] += acc(logits, labels)

    val_loss[epoch] /= len(test_dataloader)
    val_acc[epoch] /= len(test_dataloader)

    print(f"epoch: {epoch+1}, train loss: {train_loss[epoch]:.4f}, "
          f"train acc: {train_acc[epoch]:.4f}, val loss: {val_loss[epoch]:.4f}, "
          f"val_acc: {val_acc[epoch]:.4f}")

    if best_loss - val_loss[epoch] > 0.001:
        best_loss = val_loss[epoch]
        best_model = model

torch.save(best_model, 'bert.pth')
