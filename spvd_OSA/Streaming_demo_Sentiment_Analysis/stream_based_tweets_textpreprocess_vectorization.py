# -*- coding: utf-8 -*-

#Twitter Stream as Data source
twitter_stream= []
tweets = []
label = []
all_tweets = open("twitter_140.txt")  
for twts in all_tweets:  
    tweets.append(twts.replace('\n',''))
label = [0]*200+[4]*200
for i in range(len(tweets)):
    twitter_stream.append((tweets[i],label[i]))   
    
'''
Data form:
tuple:(tweet,label)
'''
    
# necessary import for Flink environment
from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import StreamingFileSink    
    
# vectorization function
def vectorization(text,label):
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
    
    # TF-IDF
    countvector    = CountVectorizer(stop_words= stopwords.words('english')) 
    tfidftransform = TfidfTransformer()
    X = countvector.fit_transform([text]).toarray() 
    X_train = tfidftransform.fit_transform(X).toarray()
    
    # return tuple of vectorized train_data and label
    return (X_train,label)

# text preprocessing/cleaning
def textprocess(text,label):
    # using regular formulation
    import re
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'^br$', ' ', text)
    text = re.sub(r'\s+^br$\s+', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'^b\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    text = text.lower().strip()
    vectorization(text,label)

# create streaming enviroment of Flink
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

# data from Twitter Stream
ds = env.from_collection(
    collection=twitter_stream,
    type_info=Types.ROW([Types.STRING(), Types.INT()]))
ds.map(lambda x: textprocess(x[0],x[1]))\
  .print()

# exectue job
env.execute("demo_clean_vectorization_job")
