# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch
import pandas as pd


from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

import config

from other_exp.utils import get_max_len, preprocess, acc

device = torch.device("cuda")


def train(batch_size=64, epochs=3, lr=5e-6, name='bert'):

    df = pd.read_csv(config.TRAIN_DATA, names=['id', 'label', 'review'])
    labels, review = df.label.values, df.review.values

    # max_len = get_max_len(review) # just checked if max len is less than 512.. but its large so
    # truncated to 512.. so no need to run this always.
    input_ids, attention_masks = preprocess(review)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_df, test_df = train_test_split(
        dataset, test_size=0.2, random_state=42, shuffle=True)

    train_dataloader = DataLoader(
        train_df, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_df, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2, output_attentions=False, output_hidden_states=False)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_loss = 1e5
    train_loss = [0] * epochs
    train_acc = [0] * epochs
    val_loss = [0] * epochs
    val_acc = [0] * epochs

    for epoch in range(0, epochs):
        model.train()

        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()

            (loss, logits) = model(input_ids,
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

    torch.save(best_model, name + '.pth')
