'''
This version concludes the following functions:
1) model splitting and merging
2) LRU strategy (now I have not consider the capacity of Redis, now the lru table is infinite increasing)
   Considering Capacity C, it should "clean" old words once the size larger than C
3) Streaming indexing
4) Temporal-sliding window
'''
from time import time
from re import sub
import re
from numpy.lib.type_check import real
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import redis
import pickle
import logging
import dill
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
#from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
#from pyflink.datastream.connectors import StreamingFileSink

class unsupervised_OSA(MapFunction):

    def __init__(self):
        self.model_param = {'min_count':2,'window_size':4,'vector_size':300}
        self.trend = 0.5
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.collector_size = 5000
        self.thousand_text = []
        self.bad,self.good = 0,0
        self.predictions = []
        self.preprocess_time = 0
        self.redis_time = 0
        self.predict_time = 0
        self.current_vocabulary = {}
        self.current_collector = []
        self.current_duplicate_word = []
        self.current_new_word = []
        self.stop_words = stopwords.words('english')
        for w in ['!',',','.','?','-s','-ly','</s>','s']:
            self.stop_words.append(w)

    def open(self, runtime_context: RuntimeContext):

        # data for initial model
        initial_tweets = pd.read_csv("flinkdata.csv")
        tweets_to_clean = initial_tweets.copy()
        tweets_to_clean = tweets_to_clean[tweets_to_clean.tweet.str.len()>1]
        tweets_to_clean.tweet = tweets_to_clean.tweet.apply(lambda x: self.text_to_word_list(x))
        initial_data = tweets_to_clean.copy()
        initial_data = initial_data[initial_data.tweet.str.len()>1]
        print("initial data is ready")

        # initial model training
        self.initial_model = Word2Vec(list(initial_data.tweet),  min_count=3,
                                                                 window=4,
                                                                 vector_size = 300,
                                                                 alpha=0.03,
                                                                 min_alpha=0.0007)
        self.corpus_count = self.initial_model.corpus_count
        self.epochs = self.initial_model.epochs

        self.index_table = range(self.initial_model.wv.index_to_key)
        self.max_index = max(self.index_table)
        self.thread_index = runtime_context.get_index_of_this_subtask()
        self.number_of_thread = runtime_context.get_number_of_parallel_subtasks()
        # save model to redis
        self.save_model(self.initial_model)

        # initial test data for evalaution
        self.test_data = pd.read_csv('osatest.csv')
        self.clean_test_data = self.test_data.tweet.apply(lambda x: self.text_to_word_list(x))
        

    def save_model(self,model):
        
        try:
            self.redis_param.set('osamodel', pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) 
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to save model to Redis server, please check your model')

    def load_model(self):

        try:
            called_model = pickle.loads(self.redis_param.get('osamodel'))
            return called_model
        except TypeError:
            logging.info('The model name you entered cannot be found in redis')
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to call the model from Redis server, please check your model')

    def tweets_collector(self,clean_words):
        result = None
        for word in clean_words:
            self.current_collector.append(word)
            
        if len(self.thousand_text) <= self.collector_size:
            self.thousand_text.append(clean_words)
        else:
            weight_matrix, to_merge = self.load_vectors()
            result = self.update_model(self.thousand_text,weight_matrix,to_merge)
            self.thousand_text = []
        if result != None:
            return result
        else:
            return 'collecting'

    
    def load_vectors(self):

        self.current_new_word = []
        self.current_duplicate_word = []
        duplicate_word_vectors = []
        index_to_call = []

        # first call the whole vocabulary
        # all_vocabulary_table = self.load_quick_query_vocabulary()
        vocabulary = self.load_vocabulary()
        self.current_vocabulary = vocabulary
        vocabulary_size = len(vocabulary)

        # call the LRU_table
        lru_table = self.load_lru_table()

        # to justify which words are new and which words are duplicate after min_count filtering
        # the words in current_collector are in order, therefore the lru table is reasonable to be updated here
        for word in self.current_collector:
            if self.current_collector.count(word) >= self.model_param['min_count']:
                if word not in vocabulary: # new words
                    lru_table.insert(0,self.max_index+1)
                    self.max_index += 1
                    self.current_new_word.append(word)
                else:                      # duplicate words
                    old_word_index = vocabulary[word]
                    lru_table.remove(old_word_index)
                    lru_table.insert(0,old_word_index)
                    self.current_duplicate_word.append(word)
        
        # call the corresponding vectors from redis
        for word in self.current_duplicate_word:
            index_to_call.append(vocabulary[word])
            duplicate_word_vectors.append(list(map(lambda x: int(x),self.redis_param.get(word)
                                          .decode()
                                          .replace('[','')
                                          .replace(']','')
                                          .split(','))))
        to_merge = (index_to_call,lru_table)
        # next: build the new initial weight matrix
        weight_matrix = self.build_weight_matrix(duplicate_word_vectors,
                                        self.model_param['vector_size'],
                                        vocabulary_size,
                                        self.current_new_word,
                                        index_to_call)
        return weight_matrix,to_merge

    def build_weight_matrix(self,duplicate_word_vectors,vector_size,size_of_vocabulary,new_words,index_to_call):
        weight_matrix = np.zeros((size_of_vocabulary, vector_size), dtype=real)
        # replace the duplicate words to weight matrix
        for index in index_to_call:
            weight_matrix[index] = duplicate_word_vectors[index] 
        weight_matrix = np.vstack([weight_matrix, np.zeros((len(new_words), vector_size), dtype=real)])
        return weight_matrix    
        
    
    def load_quick_query_vocabulary(self):
        try:
            all_vocabulary_table = self.redis_param.get('all_vocabulary')
            return all_vocabulary_table
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to load quick-query vocabulary from Redis server, please whether it exists')
            return -1

    def load_vocabulary(self):
        try:
            vocabulary = self.redis_param.get('vocabulary')
            return vocabulary
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to load vocabulary from Redis server, please whether it exists')
            return -1


    def load_lru_table(self):
        try:
            lru_table = self.redis_param.get('lru_index')
            return lru_table
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to load vocabulary from Redis server, please whether it exists')
            return -1

    # tweet preprocessing
    def text_to_word_list(self,tweet):
        
        tweet = tweet.lower() #change to lower case
        tweet = re.sub("@\w+ ","", tweet) #removes all usernames in text
        tweet = re.sub("\'s", " ", tweet) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
        tweet = re.sub("\'t", " not ", tweet)
        tweet = re.sub(" whats ", " what is ", tweet, flags=re.IGNORECASE)
        tweet = re.sub("\'ve", " have ", tweet)
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)  #remove numbers
        tweet = re.sub(r'\s+', ' ', tweet)      
        tweet = re.sub(r'[^\w\s]','',tweet)     #remove commas
        tweet = re.sub('(?<=[0-9])\,(?=[0-9])', "", tweet) # remove comma between numbers, i.e. 15,000 -> 15000
        tweet = re.sub("[!~#$+%*:()'?-]", ' ', tweet)   # remove characters stated below
        clean_word_list = tweet.split(' ')
        clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
        return clean_word_list

    
    def map(self,tweet):
        
        start = time()
        clean_word_list = self.text_to_word_list(tweet)
        try:
            clean_word_list.remove('')
            self.preprocess_time += (time() - start) / 60
            return self.tweets_collector(clean_word_list) 
        except:
            self.preprocess_time += (time() - start) / 60
            return self.tweets_collector(clean_word_list) 


    def update_model(self,new_sentences,weight_matrix, to_merge):
        
        duplicate_word_to_merge = {}
        new_word_to_merge = []
         
        model = Word2Vec(min_count   = self.model_param['min_count'],
                         window      = self.model_param['window_size'],
                         vector_size = self.model_param['vector_size'])
        model.wv.initialize_weight_matrix(weight_matrix)
        model.build_vocab(new_sentences,update = True)
        model.train(new_sentences,
                    total_examples = self.corpus_count,
                    epochs = self.epochs)
        new_weight_matrix = model.wv.get_vectors()
        for i in range(len(to_merge[0])):
            duplicate_word_to_merge[self.current_duplicate_word[i]] = new_weight_matrix[to_merge[0][i]]
        for word in self.current_new_words:
            new_word_to_merge.append([new_weight_matrix[model.key_to_index[word]],word])
        word_to_merge = (to_merge,duplicate_word_to_merge, new_word_to_merge, self.thread_index)
        # Text Vectorization is ready, next we do classifying
        ans = self.eval(new_sentences,model,word_to_merge)
        return ans

    def to_string(self,l):
        return ','.join(l)

    def tostring(self,st):
        return str(st)

    def let_round(self,list1):
        for i in range(len(list1)):
            list1[i] = round(list1[1],1)
        return list1
    
    def j_sim(self,list1,list2):
        s1 = set(list1)
        s2 = set(list2)
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))

    def haming(self,list1,list2):
        s = 0
        d = 0
        for i in range(len(list1)):
            if list1[i] == list2[i]:
                s +=1
            else:
                d +=1
        return s/d
    
    def euclidean_distance(self,vec1,vec2):
        tmp = 0
        for i in range(len(vec1)):
            tmp += (vec1[i] - vec2[i])**2
        return tmp**0.5


    def cosine_similarity(self,x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), f"{len(x)} != {len(y)}"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        # method 1
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


    def eval(self,tweets,model,word_to_merge):
        '''
        TO DO:
        Please try checking the similarity directly with sentences instead of each word
        # bad =  float(sum(model.wv.similarity(tweet,'bad')))
        # good = float(sum(model.wv.similarity(tweet,'good'))) 
        '''
        pre_time = time()
        if self.trend > 0:
            pos_coefficient = self.trend
            neg_coefficient = 1 - self.trend

        for tweet in tweets:
            self.bad,self.good = 0,0
            for words in tweet:
                try:
                    self.bad   +=  model.wv.similarity(words,'bad')   * neg_coefficient
                    self.good  +=  model.wv.similarity(words,'good')  * pos_coefficient
                except:
                    self.good += 0
        # the original labels: 0 means Negative while 4 means Positive
            if self.bad >= self.good:
                self.predictions.append(0)
            else:
                self.predictions.append(4)

        # detect trend for next collection
        self.trend = self.cosine_similarity(self.predictions,[4]*self.collector_size)

        # This version cancels the prediction results of coming tweets list(self.predictions)
        self.predict_time += (time() - pre_time) / 60

        # predict_result of test data
        predict_result = self.clean_test_data.apply(lambda x: self.predict_similarity(x,model))
        accuracy =   accuracy_score(self.test_data.polarity, list(predict_result))
        polarity = ['============'] + [self.redis_time,self.preprocess_time,self.predict_time] + [accuracy] + ['============']
        predict_polarity = self.predictions
        self.predictions = []
         
        output = (predict_polarity,word_to_merge)
    
        return output

    def predict_similarity(self,tweet,model):
        bad,good = 0,0
        for words in tweet:
            try:
                bad += model.wv.similarity(words,'bad') # * abs(self.euclidean_distance(list(model.wv.get_vectors()['bad']),list(model.wv.get_vectors()[words]))) #*self.haming(self.let_round(list(model.wv.get_vectors()[words])),bad_ref)
                good += model.wv.similarity(words,'good') # * abs(self.euclidean_distance(list(model.wv.get_vectors()['good']),list(model.wv.get_vectors()[words]))) #*self.haming(self.let_round(list(model.wv.get_vectors()[words])),good_ref)
            except:
                good += 0
        if bad > good:
            return 0
        else:
            return 4
    
    def merge(self,output1, output2):
        # form of output: (predictions, word_to_merge)
        # form: (to_merge ((index_to_call,lru_table)),duplicate_word_to_merge, new_word_to_merge)
        if len(output1) > 1:
            prediction_1 = output1[0]
            prediction_2 = output2[0]
            word_to_merge_1 = output1[1]
            word_to_merge_2 = output2[1]
            
            # release all necessary data
            index_to_call_1 = word_to_merge_1[0][0]
            index_to_call_2 = word_to_merge_2[0][0]
            lru_table_1 = word_to_merge_1[0][1]
            lru_table_2 = word_to_merge_2[0][1]
            duplicate_word_to_merge_1 = word_to_merge_1[1]
            duplicate_word_to_merge_2 = word_to_merge_2[1]
            new_word_to_merge_1 = word_to_merge_1[2]
            new_word_to_merge_2 = word_to_merge_2[2] 
            
            word_to_cover = {}
            word_to_add = []
            same_new = []
            # merge duplicate words
            # the index to call list contain absolutely the duplicate words
            # if there exists the same words of two threads

#             same_duplicate = [x for x in index_to_call_1 if x in index_to_call_2]
#             if len(same_duplicate) >=1:
#                 for same_word in same_duplicate:
#                     index_1 = index_to_call_1.index(same_word)
#                     index_2 = index_to_call_2.index(same_word)
#                     word_to_cover.append([(duplicate_word_to_merge_1[0][index_1]+duplicate_word_to_merge_2[0][index_2])/2,same_word])
            
#             l2_word = [x for x in index_to_call_2 if x not in index_to_call_1] # in list2 not in list1
#             l1_word = [x for x in index_to_call_1 if x not in index_to_call_2]
#             for word in l2_word:
#                 word_to_cover.append(duplicate_word_to_merge_2[0][index_to_call_2.index(word)],duplicate_word_to_merge_2[1][index_to_call_2.index(word)])
#             for word in l1_word:
#                 word_to_cover.append(duplicate_word_to_merge_1[0][index_to_call_1.index(word)],word)

            for key in duplicate_word_to_merge_1:
                if duplicate_word_to_merge_2.get(key):
                    word_to_cover[key]= (duplicate_word_to_merge_1[key]+duplicate_word_to_merge_2[key])/2
                else:
                    word_to_cover[key]=duplicate_word_to_merge_1[key]
            for key in duplicate_word_to_merge_2:
                if duplicate_word_to_merge_1.get(key):
                    pass
                else:
                    word_to_cover[key]=duplicate_word_to_merge_2[key]

            # cover the duplicate vector to redis
            for key in word_to_cover:
               self.redis_param.getset(key,word_to_cover.get(key))
                  
                  
            # merging the same new words
            for word in new_word_to_merge_1:
                for words in new_word_to_merge_2:
                    if word[1] == words[1]:
                        word[0] = (word[0]+words[0])/2
                        same_new.append(word[1])
            # the rest new words should be updated
            for word in new_word_to_merge_1:
                word_to_add.append(word)
            for word in new_word_to_merge_2:
                if word[1] not in same_new:
                    word_to_add.append(word)

            # save the new vector to redis
            for word in word_to_add:
                self.redis_param.set(word[1],word[0])


if __name__ == '__main__':

    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder
    coming_tweets = pd.read_csv('flinktestdata.csv')
    coming_tweets = list(coming_tweets.tweet)[180000:220000]
    print('Coming tweets is ready...')
    print('===============================')
 
    # final = []
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    
    ds = env.from_collection(collection = coming_tweets)

    ds.map(unsupervised_OSA(), output_type = Types.STRING())\
      .key_by(lambda x: x[0]).reduce(lambda x,y: unsupervised_OSA().merge(x,y))\
      .add_sink(StreamingFileSink
      .for_row_format('./output', Encoder.simple_string_encoder())
      .build())

    env.execute("osa_job")
