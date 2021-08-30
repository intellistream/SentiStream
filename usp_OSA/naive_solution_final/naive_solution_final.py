from time import time
from re import sub
import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import redis
import pickle
import logging
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import StreamExecutionEnvironment


class unsupervised_OSA(MapFunction):

    def __init__(self):
        self.time_finish = [time()]
        self.acc_to_plot = []
        self.true_label = []
        self.thousand_text = []
        self.bad, self.good = 0, 0
        self.predictions = []
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

    # tweet preprocessing
    def text_to_word_list(self, tweet):
        tweet = tweet.lower()  # change to lower case
        tweet = re.sub("@\w+ ", "", tweet)  # removes all usernames in text
        tweet = re.sub("\'s", " ",tweet)
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

        call_model = self.load_model()

        # incremental learning
        call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
        call_model.train(new_sentences,  # 2) incremental training
                         total_examples=call_model.corpus_count,
                         epochs=self.initial_model.epochs)

        self.save_model(call_model)

        # After updating model, the next step is to do classifying(predictions)
        classify_result = self.eval(new_sentences, call_model)
        return classify_result

    def eval(self, tweets, model):

        for tweet in tweets:
            for words in tweet:
                try:
                    self.bad += model.wv.similarity(words, 'bad')
                    self.good += model.wv.similarity(words, 'good')
                except:
                    self.good += 0
            # the original labels: 0 means Negative while 4 means Positive
            if self.bad >= self.good:
                self.predictions.append(1)
            else:
                self.predictions.append(2)

        self.time_finish.append(time())
        self.acc_to_plot.append(accuracy_score(self.true_label, self.predictions))
        self.predictions = []
        self.bad, self.good = 0, 0

        return str(self.acc_to_plot) + '@' + str(self.time_finish)


if __name__ == '__main__':
    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder
    import sys
    import pandas as pd
    parallelism = int(sys.argv[1])
    dataset = str(sys.argv[2])
    if dataset == 'tweet':
        # format of input data: (tweet,label)
        data = pd.read_csv('./sentiment140.csv', encoding='ISO-8859-1')
        first = data.columns[5]
        data.columns = ['polarity', 'id', 'date', 'query', 'name', 'tweet']
        tweet = list(data.tweet)
        tweet.append(first)
        label = list(data.polarity)
        label.append('0')
        data_stream = [0] * 1600000
        for i in range(len(tweet)):
            data_stream[i] = (tweet[i], int(label[i]))
    elif dataset == 'yelp':
        f = pd.read_csv('./train.csv')
        l = 60000
        first = f.columns[1]
        f.columns = ['polarity', 'tweet']
        true_label = list(f.polarity)[:l - 1]
        true_label.append('1')
        yelp_review = list(f.tweet)[:l - 1]
        yelp_review.append(first)
        data_stream = []
        for i in range(len(yelp_review)):  # len(true_label)
            data_stream.append((yelp_review[i], int(true_label[i])))

    print('Coming tweets is ready...')
    print('===============================')

    from time import time

    env = StreamExecutionEnvironment.get_execution_environment()
        # set_python_requirements(requirements_file_path='./requirements.txt',
        #                         requirements_cache_dir='cached_dir'). \
    env.set_parallelism(parallelism)

    process_time = time()
    ds = env.from_collection(collection=data_stream)
    ds.map(unsupervised_OSA(), output_type=Types.STRING()) \
        .add_sink(StreamingFileSink
                  .for_row_format('./output', Encoder.simple_string_encoder())
                  .build())
    env.execute("osa_job")
