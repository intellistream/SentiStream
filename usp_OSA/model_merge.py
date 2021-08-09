from time import time
from re import sub
import copy
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
        self.counter = 1
        self.model_to_train = None
        self.time_to_reset = 50  # reset model in redis every 100 seconds
        self.timer = time()
        self.true_label = []
        self.thousand_text = []
        self.bad, self.good = 0, 0
        self.predictions = []
        self.preprocess_time = 0
        self.redis_time = 0
        self.predict_time = 0
        self.collector_time = 0
        self.training_time = 0
        self.stop_words = stopwords.words('english')
        self.collector_size = 2000
        for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
            self.stop_words.append(w)

    def open(self, runtime_context: RuntimeContext):
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
        # ar = open('./amazon.txt', 'r')
        # self.true_label = []  # 1 neg
        # for review in ar:
        #     if '__label__2' in review:
        #         self.true_label.append(4)
        #     elif '__label__1' in review:
        #         self.true_label.append(0)

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
        if len(self.thousand_text) >= 2000:
            ans = self.update_model(self.thousand_text)
            self.thousand_text = []
            self.true_label = []
            return ans
        else:
            ans = ('collecting', 1)
            return ans

    def map(self, tweet):

        self.true_label.append(tweet[1])
        return self.text_to_word_list(tweet[0])

    def get_model_new(self, final_words, final_vectors, final_syn1, final_syn1neg, final_cum_table, corpus_count,
                      final_count, final_sample_int, final_code, final_point, model):

        model_new = copy.deepcopy(model)
        n_words = len(final_words)
        model_new.wv.index_to_key = final_words
        model_new.wv.key_to_index = {word: idx for idx, word in enumerate(final_words)}
        model_new.wv.vectors = final_vectors
        model_new.syn1 = final_syn1
        model_new.syn1neg = final_syn1neg
        model_new.syn1 = final_syn1
        model_new.syn1neg = final_syn1neg
        model_new.cum_table = final_cum_table
        # model3.syn0_lockf = np.array([1. for _ in range(n_words)])
        model_new.corpus_count = corpus_count
        model_new.corpus_total_words = n_words
        model_new.wv.expandos['count'] = final_count
        model_new.wv.expandos['sample_int'] = final_sample_int
        model_new.wv.expandos['code'] = final_code
        model_new.wv.expandos['point'] = final_point
        return model_new

    def update_model(self, new_sentences):

        '''
        To Do: not total model needed, call the correspoding word vectors only
        method: 1) query 2) extract
        '''
        # start_load_redis = time()
        if self.counter == 1:
            call_model = self.load_model()
            self.counter += 1
        else:
            call_model = self.model_to_train
        # end_load_redis = time()

        # time of loading model from Redis
        # loadtime = (end_load_redis - start_load_redis) / 60

        # incremental learning
        # learn_start_time = time()
        call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
        call_model.train(new_sentences,  # 2) incremental training
                         total_examples=call_model.corpus_count,
                         epochs=self.initial_model.epochs)
        # self.training_time += (time() - learn_start_time) / 60

        # start_save_redis = time()
        # self.save_model(call_model)
        # end_save_redis = time()
        #
        # # time of saving model from Redis
        # savetime = (end_save_redis - start_save_redis) / 60
        #
        # # total time cost of Redis
        # self.redis_time += (loadtime + savetime)
        self.model_to_train = call_model
        # After updating model, the next step is to do classifyings(predictions)
        classify_result = self.eval(new_sentences, call_model)
        return classify_result

    # ======== these two functions are used for output  ========
    def to_string(self, l):
        return ','.join(l)

    def tostring(self, st):
        return str(st)

    # ==========================================================

    def model_merge(self, model_and_acc1, model_and_acc2):
        if model_and_acc1[0] == 'collecting':
            return model_and_acc1[1]+model_and_acc2[1]
        elif model_and_acc1[0] == 'acc':
            return model_and_acc1[1]+' '+model_and_acc2[1]
        else:

            model1 = model_and_acc1[1]
            model2 = model_and_acc2[1]
            # acc1 = model_and_acc1[1]
            # acc2 = model_and_acc2[1]
            words1 = copy.deepcopy(model1.wv.index_to_key)
            words2 = copy.deepcopy(model2.wv.index_to_key)
            syn1s1 = copy.deepcopy(model1.syn1)
            syn1s2 = copy.deepcopy(model2.syn1)
            syn1negs1 = copy.deepcopy(model1.syn1neg)
            syn1negs2 = copy.deepcopy(model2.syn1neg)
            cum_tables1 = copy.deepcopy(model1.cum_table)
            cum_tables2 = copy.deepcopy(model2.cum_table)
            corpus_count = copy.deepcopy(model1.corpus_count) + copy.deepcopy(model2.corpus_count)
            counts1 = copy.deepcopy(model1.wv.expandos['count'])
            counts2 = copy.deepcopy(model2.wv.expandos['count'])
            sample_ints1 = copy.deepcopy(model1.wv.expandos['sample_int'])
            sample_ints2 = copy.deepcopy(model2.wv.expandos['sample_int'])
            codes1 = copy.deepcopy(model1.wv.expandos['code'])
            codes2 = copy.deepcopy(model2.wv.expandos['code'])
            points1 = copy.deepcopy(model1.wv.expandos['point'])
            points2 = copy.deepcopy(model2.wv.expandos['point'])
            final_words = []
            final_vectors = []
            final_syn1 = []
            final_syn1neg = []
            final_cum_table = []
            final_count = []
            final_sample_int = []
            final_code = []
            final_point = []
            for idx1 in range(len(words1)):
                word = words1[idx1]
                '''
                if word == '现状':
                    print(model1.wv.vocab['现状'].index)
                '''
                v1 = model1.wv[word]
                syn11 = syn1s1[idx1]
                syn1neg1 = syn1negs1[idx1]
                cum_table1 = cum_tables1[idx1]
                count = counts1[idx1]
                sample_int = sample_ints1[idx1]
                code = codes1[idx1]
                point = points1[idx1]
                try:
                    idx2 = words2.index(word)
                    v2 = model2.wv[word]
                    syn12 = syn1s2[idx2]
                    syn1neg2 = syn1negs2[idx2]
                    cum_table2 = cum_tables2[idx2]
                    v = np.mean(np.array([v1, v2]), axis=0)
                    syn1 = np.mean(np.array([syn11, syn12]), axis=0)
                    syn1neg = np.mean(np.array([syn1neg1, syn1neg2]), axis=0)
                    cum_table = np.mean(np.array([cum_table1, cum_table2]), axis=0)
                except:
                    v = v1
                    syn1 = syn11
                    syn1neg = syn1neg1
                    cum_table = cum_table1
                final_words.append(word)
                final_vectors.append(list(v))
                final_syn1.append(syn1)
                final_syn1neg.append(syn1neg)
                final_cum_table.append(cum_table)
                final_count.append(count)
                final_sample_int.append(sample_int)
                final_code.append(code)
                final_point.append(point)
            for idx2 in range(len(words2)):
                word = words2[idx2]
                if word in final_words:
                    continue
                v2 = model2.wv[word]
                syn12 = syn1s2[idx2]
                syn1neg2 = syn1negs2[idx2]
                cum_table2 = cum_tables2[idx2]
                count = counts2[idx2]
                sample_int = sample_ints2[idx2]
                code = codes2[idx2]
                point = points2[idx2]
                try:
                    idx1 = words1.index(word)
                    v1 = model1.wv[word]
                    syn11 = syn1s1[idx1]
                    syn1neg1 = syn1negs1[idx1]
                    cum_table1 = cum_tables1[idx1]
                    v = np.mean(np.array([v1, v2]), axis=0)
                    syn1 = np.mean(np.array([syn11, syn12]), axis=0)
                    syn1neg = np.mean(np.array([syn1neg1, syn1neg2]), axis=0)
                    cum_table = np.mean(np.array([cum_table1, cum_table2]), axis=0)
                except:
                    v = v2
                    syn1 = syn12
                    syn1neg = syn1neg2
                    cum_table = cum_table2
                final_words.append(word)
                final_vectors.append(list(v))
                final_syn1.append(syn1)
                final_syn1neg.append(syn1neg)
                final_cum_table.append(cum_table)
                final_count.append(count)
                final_sample_int.append(sample_int)
                final_code.append(code)
                final_point.append(point)

            model_new = self.get_model_new(final_words, np.array(final_vectors), np.array(final_syn1),
                                           np.array(final_syn1neg), \
                                           final_cum_table, corpus_count, np.array(final_count), np.array(final_sample_int), \
                                           np.array(final_code), np.array(final_point), model1)
            self.save_model(model_new)
            self.counter = 1
            return model_new

    def eval(self, tweets, model):

        # pre_time = time()

        for tweet in tweets:
            good, bad = 0, 0
            for words in tweet:
                try:
                    bad += model.wv.similarity(words, 'bad')
                    good += model.wv.similarity(words, 'good')
                    bad += model.wv.similarity(words, 'low')
                    good += model.wv.similarity(words, 'high')
                    bad += model.wv.similarity(words, 'fuck')
                    good += model.wv.similarity(words, 'thank')
                    bad += model.wv.similarity(words, 'poor')
                    good += model.wv.similarity(words, 'great')
                    # bad += model.wv.similarity(words, 'hard')
                    # good += model.wv.similarity(words, 'easy')
                    # bad += model.wv.similarity(words, 'wrong')
                    # good += model.wv.similarity(words, 'right')
                    # bad += model.wv.similarity(words, 'horrible')
                    # good += model.wv.similarity(words, 'amazing')
                except:
                    good += 0
            # the original labels: 0 means Negative while 4 means Positive
            if bad >= good:
                self.predictions.append(1)
            else:
                self.predictions.append(2)

        # This version cancels the prediction results of coming tweets list(self.predictions)
        # self.predict_time += (time() - pre_time) / 60
        acc = str(accuracy_score(self.predictions, self.true_label))
        self.predictions = []
        if time() - self.timer >= self.time_to_reset:
            model_to_merge = ('model', model)
            # predict_result of test data
            # self.timer = time()
            self.timer = time()
            return model_to_merge
        else:
            not_yet = ('acc', acc)
            # predict_result of test data
            return not_yet


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
    true_label = list(f.polarity)[:80000]
    yelp_review = list(f.tweet)[:80000]
    yelp_stream = []
    for i in range(80000):
        yelp_stream.append((yelp_review[i], true_label[i]))

    print('Coming tweets is ready...')
    print('===============================')
    # '0.473 0.5765 0.5945 0.5495 0.5525 0.626 0.6135 0.576 0.654 0.665 0.667 0.5405 0.536 0.481 0.698 0.622 0.633 0.721 0.63 0.58 0.658 0.692 0.8065 0.759
    # final = []
    # '0.5115 0.6445 0.5955 0.5185 0.483 0.683 0.688 0.6835 0.6945 0.6875 0.671 0.6805 0.627 0.66 0.575 0.803 0.676 0.754 0.6155 0.677 0.671 0.5845 0.583')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)

    ds = env.from_collection(collection=yelp_stream)
    # ds.shuffle()
    ds.map(unsupervised_OSA()) \
        .key_by(lambda x: x[0]) \
        .reduce(lambda x, y: (x[0], unsupervised_OSA().model_merge(x, y))) \
        .print()

    # .add_sink(StreamingFileSink
    #           .for_row_format('./output', Encoder.simple_string_encoder())
    #           .build())

    env.execute("osa_job")
