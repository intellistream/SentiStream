# pylint: disable=import-error
from transformers import BertTokenizer
import torch
import re
import string


NEGATION_WORDS = {'not', 'no', 'didn', 'didnt', 'wont',
                  'dont', 'don', 'doesnt', 'doesn', 'shouldnt', 'shouldn'}

STOP_WORDS = {'also', 'ltd', 'once', 'll', 'make', 'he', 'through', 'all', 'top', 'from', 'or', 's',
              'hereby', 'so',  'yours', 'since', 'meanwhile', 're', 'over', 'mrs', 'thereafter',
              'ca', 'move', 'mill', 'such', 'wherever', 'on', 'besides', 'few', 'does', 'yet',  'y',
              'much', 'my', 'him', 'yourselves', 'as', 'ours', 'therefore', 'amongst', 'due', 'mr',
              'here', 'may', 'onto', 'it', 'whose', 'himself', 'least', 'i', 'what', 'many', 'd',
              'hereafter', 'anything', 'of', 'whoever', 'made', 'be', 'sometimes', 'put', 'found',
              'than', 'although', 'anyway', 'seems', 'you', 'under', 'above', 'themselves', 'thus',
              'a', 'con', 'when', 'why', 'back', 'until', 'first', 'theirs', 'describe', 'because',
              'always', 'too', 'across', 't', 'anyhow', 'her', 'ourselves', 'latterly', 'six', 'an',
              'somewhere', 'else', 'for', 'really', 'up', 'among', 'used', 'whenever', 'during',
              'nowhere', 'nothing', 'if', 'afterwards', 'that', 'whereas', 'elsewhere', 'along',
              'been', 'both', 'etc', 'ie', 'might', 'into', 'inc', 'with', 'formerly', 'there',
              'will', 'own', 'seemed', 'though', 'was', 'whereupon', 'just', 'except', 'has',
              'your', 'do', 'around', 'herein', 'anywhere', 'rd',  'now', 'sincere', 'this',  'me',
              'throughout',  'unless', 'against', 'out', 'most', 'various', 'others', 'them', 'th',
              'eleven', 'am', 'indeed', 'name', 'his', 'often', 'yourself', 'only', 'kg', 'take',
              'everything', 'cry', 'and',  'quite', 'itself', 'in', 'to', 'well', 'namely', 'thru',
              'see', 'would', 'which', 'beforehand', 'myself', 'having', 'however', 'go', 'did',
              'below', 'those', 'st', 'computer', 'several', 'whether', 'have', 'between', 'any',
              'becoming', 'thereby', 'while', 'were', 'whole', 'latter', 'but', 'km', 'amount',
              'either', 'herself', 'whereafter', 'never', 'system', 'un', 'find', 'please', 'o',
              'hereupon', 'thin', 'give', 'third', 'every', 'doing', 'our', 'towards', 'another',
              'before', 'within', 'mine', 'almost', 'mostly', 'down', 'de', 'seeming', 'moreover',
              'some', 'us', 'former', 'call', 'should', 'she', 'even', 'beyond', 'became', 'other',
              'show', 'eg', 'about', 'side', 'its', 'these', 'rather', 'alone', 'nd', 'after',
              'already', 'keep', 'more', 'behind', 'thick', 'together', 'upon', 'interest', 'dr',
              'otherwise', 'full', 'can', 'next', 'last', 'bill', 'their', 'hers', 'hence', 'by',
              'become', 'something', 'who', 'further', 'someone', 'must', 'say', 'each', 'very',
              'whom', 'again', 'then', 'we', 'same', 'via', 'where', 'per', 'are', 'the', 'still',
              'toward', 'anyone', 'therein', 'being', 'off', 'perhaps', 'is', 'had', 'co', 'at',
              'done', 'everywhere', 'less', 'wherein', 'could', 'ma', 'sometime', 'seem', 'somehow',
              'beside', 'whatever', 'whereby', 'ever', 'everyone', 'nevertheless', 'serious',
              'using', 'becomes', 'enough', 'how', 'bottom', 've', 'regarding', 'm', 'they', 'part',
              'front', 'fill', 'get', 'nobody', 'detail'}

url_rx = re.compile(r"http\S+|www\S+|@\w+|#\w+")
html_rx = re.compile(r'<.*?>')
multi_dot_rx = re.compile(r'\.{2,}')

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
