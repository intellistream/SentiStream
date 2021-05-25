# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# Flink
from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import StreamingFileSink

# text processing
import re

# vectorization
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# necessary import of Redis
import pickle
import redis
import logging

# call the sklearn ml model
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

# boot enviroment
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

#===================================== Code start here ===========================================#



#Twitter Stream as Data source
tweets = []
train_data = []
label = [0]*200+[4]*200
all_tweets = open("twitter_140.txt") 
for twts in all_tweets:  
    tweets.append(twts.replace('\n',''))
    
corpus = []
for i in range(0,400):
    clean_text = re.sub(r'\W', ' ', str(tweets[i]))
    clean_text = re.sub(r'^br$', ' ', clean_text)
    clean_text = re.sub(r'\s+^br$\s+', ' ', clean_text)
    clean_text = re.sub(r'\s+[a-z]\s+', ' ', clean_text)
    clean_text = re.sub(r'^b\s+', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text) 
    clean_text = clean_text.lower()    
    corpus.append(clean_text)  

countvector    = CountVectorizer(stop_words= stopwords.words('english')) 
tfidftransform = TfidfTransformer()
X = countvector.fit_transform(corpus).toarray() 
X_train = tfidftransform.fit_transform(X).toarray()

for i in range(len(tweets)):
    train_data.append((list(X_train[i]),label[i]))
'''
Data Form:
tuple:(Vectorized_train_data, label)
'''
# Naive Bayes model
# pre-trained model to avoid 'cold start'
BNB = BernoulliNB()
BNB.fit([train_data[0][0],train_data[1][0]],label[:2])
# save the model to redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)
try:
    r.set('osamodel', pickle.dumps(BNB, protocol=pickle.HIGHEST_PROTOCOL))
    print('model initialized')
except (redis.exceptions.RedisError, TypeError, Exception):
    logging.warning('unable to get model from redis server, please check')

    
# incremental SA model training 
def train_predict(X_train,label):
    
    # call the initialized model from redis
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    # load the model
    called_model = pickle.loads(r.get('osamodel'))
    
    # incremental learnning from coming data
    called_model.partial_fit([X_train],[int(label)])
    
    # save model to Redis
    try:
        r.set('osamodel', pickle.dumps(called_model, protocol=pickle.HIGHEST_PROTOCOL))
        print('model initialized')
    except (redis.exceptions.RedisError, TypeError, Exception):
        logging.warning('unable to get model from redis server, please check')
        
    # show the result on terminal
    a ='model updated'
    return a

# create flink streaming enviroment
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)
ds = env.from_collection(collection=train_data) #    type_info=Types.ROW([Types.PRIMITIVE_ARRAY(Types.DOUBLE()), Types.INT()])

# print results in terminal
ds.map(lambda x: train_predict(x[0],x[1]))\
  .print()

# execute job
env.execute("osa_job")
