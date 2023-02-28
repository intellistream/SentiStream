import pandas as pd 

from nltk.corpus import stopwords
# import spacy 
# en = spacy.load('en_core_web_sm')


# print(stopwords.words('english'))
# print(en.Defaults.stop_words)
df = pd.read_csv('./train.csv', names=['label', 'review'])

# stop_words = ['no', 'nor', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', \
#     "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop_words = ['not']

# df = df[:100000]

# reviews = df['review'].tolist()

ngram_rev = []

for idx, row in df.iterrows():
    for stop_word in stop_words:
        if stop_word in row['review']:
            ngram_rev.append(row['review'])
        break


# for rev in ngram_rev:
#     print(rev)
#     break

print(len(ngram_rev))