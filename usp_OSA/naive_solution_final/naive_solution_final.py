from time import time
from re import sub
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import redis
import pickle
import logging
# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
# from pyflink.common.serialization import SimpleStringEncoder
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment


# from pyflink.datastream.connectors import StreamingFileSink

class unsupervised_OSA(MapFunction):

    def __init__(self):
        self.acc_to_plot = []
        self.true_label = []
        self.thousand_text = []
        self.bad, self.good = 0, 0
        self.predictions = []
        self.preprocess_time = 0
        self.redis_time = 0
        self.predict_time = 0
        self.collector_time = 0
        self.stop_words = stopwords.words('english')
        self.collector_size = 2000
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
            self.stop_words.append(w)

    def open(self, runtime_context: RuntimeContext):
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)

        # data for initial model
        self.initial_model = Word2Vec.load('ogb.model')
        # save model to redis
        self.save_model(self.initial_model)

    def save_model(self, model):
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
        try:
            self.redis_param.set('osamodel', pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to save model to Redis server, please check your model')

    def load_model(self):
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
        try:
            called_model = pickle.loads(self.redis_param.get('osamodel'))
            return called_model
        except TypeError:
            logging.info('The model name you entered cannot be found in redis')
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to call the model from Redis server, please check your model')

    '''
    To Do: another map fucntion to deal with cleaning
    '''

    # tweet preprocessing
    def text_to_word_list(self, tweet):

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
        clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
        while '' in clean_word_list:
            clean_word_list.remove('')
        self.thousand_text.append(clean_word_list)
        if len(self.thousand_text) >= self.collector_size:
            ans = self.update_model(self.thousand_text)
            self.thousand_text = []
            self.true_label = []
            return ans
        else:
            return 'collecting'

    def map(self, tweet):

        self.true_label.append(tweet[1])
        return self.text_to_word_list(tweet[0])

    def update_model(self, new_sentences):

        '''
        To Do: not total model needed, call the correspoding word vectors only
        method: 1) query 2) extract
        '''
        start_load_redis = time()
        call_model = self.load_model()
        end_load_redis = time()

        # time of loading model from Redis
        loadtime = (end_load_redis - start_load_redis) / 60

        # incremental learning
        # learn_start_time = time()
        call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
        call_model.train(new_sentences,  # 2) incremental training
                         total_examples=call_model.corpus_count,
                         epochs=self.initial_model.epochs)
        # self.training_time += (time() - learn_start_time) / 60

        start_save_redis = time()
        self.save_model(call_model)
        end_save_redis = time()
        #
        # # time of saving model from Redis
        savetime = (end_save_redis - start_save_redis) / 60
        #
        # # total time cost of Redis
        self.redis_time += (loadtime + savetime)

        # After updating model, the next step is to do classifyings(predictions)
        classify_result = self.eval(new_sentences, call_model)
        return classify_result

    # ======== these two functions are used for output  ========
    def to_string(self, l):
        return ','.join(l)

    def tostring(self, st):
        return str(st)

    # ==========================================================

    def eval(self, tweets, model):

        pre_time = time()

        for tweet in tweets:
            good, bad = 0, 0
            for words in tweet:
                try:
                    bad += model.wv.similarity(words, 'bad')
                    good += model.wv.similarity(words, 'good')
                    # bad   +=  model.wv.similarity(words,'low')
                    # good  +=  model.wv.similarity(words,'high')
                    # bad   +=  model.wv.similarity(words,'fuck')
                    # good  +=  model.wv.similarity(words,'thank')
                    # bad   +=  model.wv.similarity(words,'poor')
                    # good  +=  model.wv.similarity(words,'great')
                    # bad   +=  model.wv.similarity(words,'hard')
                    # good  +=  model.wv.similarity(words,'easy')
                    # self.bad   +=  model.wv.similarity(words,'wrong')
                    # self.good  +=  model.wv.similarity(words,'right')
                    # self.bad   +=  model.wv.similarity(words,'horrible')
                    # self.good  +=  model.wv.similarity(words,'amazing')
                except:
                    good += 0
            # the original labels: 0 means Negative while 4 means Positive
            if bad >= good:
                self.predictions.append(1)
            else:
                self.predictions.append(2)

        # This version cancels the prediction results of coming tweets list(self.predictions)
        self.predict_time += (time() - pre_time) / 60

        # predict_result of test data
        # return self.tostring(self.predictions)
        self.acc_to_plot.append(accuracy_score(self.true_label, self.predictions))

        self.predictions = []

        return str(self.acc_to_plot) + '====' + str(self.redis_time) + str(self.predict_time) + '===='


if __name__ == '__main__':
    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder

    # f = open('./amazon.txt', 'r')
    # true_label = []  # 1 neg
    # amazon_review = []  # 2 pos
    # for line in f:
    #     if '__label__2' in line:
    #         true_label.append(4)
    #         amazon_review.append(line[11:].replace('\n', ''))
    #     elif '__label__1' in line:
    #         true_label.append(0)
    #         amazon_review.append(line[11:].replace('\n', ''))
    # amazon_stream = []
    # for i in range(20000):
    #     amazon_stream.append((amazon_review[i], true_label[i]))
    f = pd.read_csv('./train.csv')
    true_label = list(f.polarity)[:20000]
    yelp_review = list(f.tweet)[:20000]
    yelp_stream = []
    for i in range(20000):
        yelp_stream.append((yelp_review[i], true_label[i]))

    print('Coming tweets is ready...')
    print('===============================')

    # final = []
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    ds = env.from_collection(collection=yelp_stream)
    # ds.shuffle()
    ds.map(unsupervised_OSA(), output_type=Types.STRING()) \
        .add_sink(StreamingFileSink
                  .for_row_format('./output', Encoder.simple_string_encoder())
                  .build())

    env.execute("osa_job")
