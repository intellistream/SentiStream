import re
import string
import gensim
import pandas as pd

from utils import tokenize, clean_for_wv

from sklearn.metrics import accuracy_score

from gensim import corpora

url_rx = re.compile(r"http\S+|www\S+|\@\w+|#\w+")
multi_dot_rx = re.compile(r'\.{2,}')
ws_rx = re.compile(r'\s+')
alpha_table = str.maketrans({char: ' ' if char not in (
    '?', '!', '.') and not char.isalpha() else char for char in string.punctuation + string.digits})


new_df = pd.read_csv('../train.csv', names=['label', 'review'])
new_df['label'] -= 1

preds = []

for idx in range(0, len(new_df), 10000):
    df = new_df.iloc[idx:idx+10000,:]

    doc_tokens = clean_for_wv([tokenize(text) for text in df.review.values])

    dictionary = corpora.Dictionary(doc_tokens)
    corpus = [dictionary.doc2bow(doc) for doc in doc_tokens]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=2, passes=20)

    y_preds = []

    for i, doc in enumerate(doc_tokens):
        topic_distribution = lda_model.get_document_topics(corpus[i])[0]
        y_preds.append(1 if topic_distribution[0] > 0.5 else 0)

    preds.append(accuracy_score(df.label.values, y_preds))
    print(preds[-1], idx)

print(preds)
