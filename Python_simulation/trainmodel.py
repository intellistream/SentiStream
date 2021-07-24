def text_to_word_list(tweet):

    tweet = tweet.lower()  # change to lower case
    tweet = re.sub("@\w+ ", "", tweet)  # removes all usernames in text
    tweet = re.sub("\'s", " ",
                   tweet)  # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    tweet = re.sub("\'t", " not ", tweet)
    tweet = re.sub(" whats ", " what is ", tweet, flags=re.IGNORECASE)
    tweet = re.sub("\'ve", " have ", tweet)
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)  # remove numbers
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)  # remove commas
    tweet = re.sub('(?<=[0-9])\,(?=[0-9])', "", tweet)  # remove comma between numbers, i.e. 15,000 -> 15000
    tweet = re.sub("[!~#$+%*:()'?-]", ' ', tweet)  # remove characters stated below
    clean_word_list = tweet.split(' ')
    clean_word_list = [w for w in clean_word_list if w not in stop_words]
    while '' in clean_word_list:
        clean_word_list.remove('')
    return clean_word_list
    
    # f = open('./amazon.txt', 'r')
# amazon_review = []  # 2 pos
# for line in f:
#     if '__label__2' in line:
#         amazon_review.append(line[11:].replace('\n', ''))
#     elif '__label__1' in line:
#         amazon_review.append(line[11:].replace('\n', ''))
# amazon_stream = []
# for i in range(20001, 160001): # len(amazon_review)
#     amazon_stream.append(amazon_review[i])

from gensim.models import Word2Vec
from re import sub
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import redis
import pickle
import logging
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
stop_words = stopwords.words('english')
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
    stop_words.append(w)
    
    
    clean_tweet =[]
for review in amazon_stream:
    clean_tweet.append(text_to_word_list(review))
    
    
    
    yelp_model = Word2Vec(min_count=3,
                         window=4,
                         vector_size=300,
                         sample =1e-5,
                         alpha=0.03, 
                         min_alpha=0.0007,
                        negative =20)
yelp_model.build_vocab(clean_review[:200000])
yelp_model.train(clean_review[:200000], total_examples = yelp_model.corpus_count, epochs = 30)


amazon_model.save('ns.model')








import pandas as pd
yelp_stream = []
yelp = pd.read_csv('./train.csv')
yelp_stream =  list(yelp.tweet)
true_label = list(yelp.polarity)
yelp_stream
clean_review =[]
for review in yelp_stream:
    clean_review.append(text_to_word_list(review))
clean_review
yelp_model = Word2Vec(min_count=3,
                         window=4,
                         vector_size=300,
                         sample =1e-5,
                         alpha=0.03, 
                         min_alpha=0.0007,
                        negative =20)
yelp_model.build_vocab(clean_review[:200000])
print(yelp_model)
yelp_model.train(clean_review[:200000], total_examples = yelp_model.corpus_count, epochs = 30)



def eval(tweets, model):
    predictions = []
    for tweet in tweets:
        bad, good = 0, 0
        for words in tweet:
            try:
                bad += model.wv.similarity(words, 'bad')
                good += model.wv.similarity(words, 'good')
            except:
                good += 0
        if bad >= good:
            predictions.append(1)
        else:
            predictions.append(2)
    return predictions
acc_to_plot = []
for i in range(1,11):
    acc= accuracy_score(eval(clean_review[(i-1)*2000:2000*i],yelp_model), true_label[(i-1)*2000:2000*i])
    acc_to_plot.append(acc)
    acc_to_plot.append(acc)
acc_to_plot
