from transformers import BertTokenizer
import torch
import re
import string


url_rx = re.compile(r"http\S+|www\S+|\@\w+")
multi_dot_rx = re.compile(r'\.{2,}')
ws_rx = re.compile(r'\s+')
alpha_table = str.maketrans({char: ' ' if char not in (
    '?', '!', '.') and not char.isalpha() else char for char in string.punctuation + string.digits})
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)


def get_max_len(review):
    max_len = 0
    for sent in review:
        input_ids = tokenizer.encode(
            sent, add_special_tokens=True, max_length=512, truncation=True)
        max_len = max(max_len, len(input_ids))
    return max_len


def preprocess(review):
    input_ids = []
    attention_masks = []
    for sent in review:
        sent = url_rx.sub('', sent).lower()
        sent = sent.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        sent = multi_dot_rx.sub('.',  sent)
        sent = sent.translate(alpha_table)
        sent = sent.replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ')

        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks


def acc(preds, labels):
    preds = torch.argmax(preds, axis=1)
    return torch.sum(preds == labels) / len(labels)
