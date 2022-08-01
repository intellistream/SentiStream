#!/usr/bin/env python3
import random
import copy
import re
import numpy as np
import argparse

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from numpy import dot
from numpy.linalg import norm
from gensim.models import Word2Vec

import redis
import pickle
import logging

import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.connectors import StreamingFileSink
from pyflink.common.serialization import Encoder

from utils import process, split

from time import time
import pandas as pd

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('plstream.log', mode='w')
formatter = logging.Formatter('PLStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

# class for_output(MapFunction):
#     def __init__(self):
#         pass
#
#     def map(self, value):
#         return str(value[1])
#
#     def logFile(self, f, m):
#         with open(f, 'a') as wr:
#             wr.write(m)

MODE = 'LABEL'


class unsupervised_OSA(MapFunction):

    def __init__(self):
        self.initial_model = None
        self.redis_param = None
        self.start_timer = time()
        # collection
        self.vocabulary = []
        self.true_label = []
        self.collector = []
        self.cleaned_text = []
        self.stop_words = stopwords.words('english')
        self.collector_size = 2000

        # model pruning
        self.LRU_index = ['good', 'bad']
        # self.max_index = max(self.LRU_index)
        self.LRU_cache_size = 300000
        # self.sno = nltk.stem.SnowballStemmer('english')

        # model merging
        self.flag = True
        self.model_to_train = None
        self.timer = time()
        self.time_to_reset = 30

        # similarity-based classification preparation
        self.true_ref_neg = []
        self.true_ref_pos = []
        self.ref_pos = ['love', 'best', 'beautiful', 'great', 'cool', 'awesome', 'wonderful', 'brilliant', 'excellent',
                        'fantastic']
        self.ref_neg = ['bad', 'worst', 'stupid', 'disappointing', 'terrible', 'rubbish', 'boring', 'awful',
                        'unwatchable', 'awkward']
        # self.ref_pos = [self.sno.stem(x) for x in self.ref_pos]
        # self.ref_neg = [self.sno.stem(x) for x in self.ref_neg]

        # temporal trend detection
        self.pos_coefficient = 0.5
        self.neg_coefficient = 0.5

        # results
        self.confidence = 0.5
        # self.acc_to_plot = []
        # self.acc_to_plot = []
        self.predictions = []
        self.labelled_dataset = []
        self.confidence_list = []

    def open(self, runtime_context: RuntimeContext):
        # redis-server parameters
        self.redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)

        # load initial model
        self.initial_model = Word2Vec.load('PLS_c10.model')
        self.vocabulary = list(self.initial_model.wv.index_to_key)

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
        # try:
        called_model = pickle.loads(self.redis_param.get('osamodel'))
        return called_model
        # except TypeError:
        #     logging.info('The model name you entered cannot be found in redis')
        # except (redis.exceptions.RedisError, TypeError, Exception):
        #     logging.warning('Unable to call the model from Redis server, please check your model')

    # tweet preprocessing
    def text_to_word_list(self, text):
        # text = re.sub("@\w+ ", "", text)
        # text = re.sub("[!~#$+%*:()'?-]", ' ', text)
        # text = re.sub('[^a-zA-Z]', ' ', text)
        # clean_word_list = text.strip().split(' ')
        # clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
        clean_word_list = process(text)
        # logging.warning("clean_word_list: " + str(clean_word_list))
        while '' in clean_word_list:
            clean_word_list.remove('')
        self.cleaned_text.append(clean_word_list)
        if len(self.cleaned_text) >= self.collector_size:
            # logger.info('text to word list update model')
            # ans = self.update_model(self.cleaned_text)
            # return ans
            return 'update_model'

    def model_prune(self, model):
        if len(model.wv.index_to_key) <= self.LRU_cache_size:
            logger.info('model prune')
            return model
        else:
            word_to_prune = list(self.LRU_index[30000:])
            for word in word_to_prune:
                k = model.wv.key_to_index[word]
                del model.wv.index_to_key[k]
                del model.wv.key_to_index[word]
            self.vocabulary = list(model.wv.index_to_key)
            return model

    def get_model_new(self, final_words, final_vectors, final_syn1, final_syn1neg, final_cum_table, corpus_count,
                      final_count, final_sample_int, final_code, final_point, model):

        model_new = copy.deepcopy(model)
        n_words = len(final_words)
        model_new.wv.index_to_key = final_words
        model_new.wv.key_to_index = {word: idx for idx, word in enumerate(final_words)}
        model_new.wv.vectors = final_vectors
        model_new.syn1 = final_syn1  # dk why this is important
        model_new.syn1neg = final_syn1neg
        model_new.syn1 = final_syn1
        model_new.syn1neg = final_syn1neg
        model_new.cum_table = final_cum_table
        model_new.corpus_count = corpus_count
        model_new.corpus_total_words = n_words
        model_new.wv.expandos['count'] = final_count
        model_new.wv.expandos['sample_int'] = final_sample_int
        model_new.wv.expandos['code'] = final_code
        model_new.wv.expandos['point'] = final_point
        return model_new

    def model_merge(self, model1, model2):
        # prediction or accuracy not merging
        logger.warning('model_merge')
        logging.warning("model merge")
        if model1[0] == 'labelled':
            logging.warning(model1)
            return (model1[1]) + (model2[1])
        elif model1[0] == 'acc':
            return (float(model1[1]) + float(model2[1])) / 2
        # actual merging taking place
        elif model1[0] == 'model':
            logger.info('model_merge model')
            model1 = model1[1]
            model2 = model2[1]
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
                                           np.array(final_syn1neg), final_cum_table, corpus_count,
                                           np.array(final_count),
                                           np.array(final_sample_int), np.array(final_code), np.array(final_point),
                                           model1)
            self.save_model(model_new)
            self.flag = True
            logging.warning("model 1 merge time: " + str(time() - model1[2]))
            logging.warning("model 2 merge time: " + str(time() - model2[2]))
            return model_new

    def map(self, tweet):
        # logger.info(tweet[0][:20] + '... ' + str(tweet[1]))
        # self.logFile('plstream.log',str(self)+'\n')
        self.true_label.append(int(tweet[1]))
        # return "ping"
        if MODE == "LABEL":
            self.collector.append((tweet[0], tweet[2]))
        tokenise_text = self.text_to_word_list(tweet[2])
        if tokenise_text == 'update model':
            return self.update_model(self.cleaned_text)

            self.cleaned_text = []
            self.true_label = []
            classify_result = self.classify_result(new_sentences, self.model_to_train)

            if time() - self.timer >= self.time_to_reset:  # prune and return model
                self.model_to_train = self.model_prune(self.model_to_train)
                model_to_merge = ('model', self.model_to_train, self.start_timer)
                self.timer = time()
                return model_to_merge
            else:
                if MODE == 'LABEL':
                    not_yet = ('labelled', classify_result)
                else:
                    not_yet = ('acc', classify_result)
                return not_yet
        else:
            return ('collecting', '1')

    def incremental_training(self, new_sentences):
        self.model_to_train.build_vocab(new_sentences, update=True)  # 1) update vocabulary
        self.model_to_train.train(new_sentences,  # 2) incremental training
                                  total_examples=self.model_to_train.corpus_count,
                                  epochs=self.model_to_train.epochs)

    def update_LRU_index(self):
        for word in self.model_to_train.wv.index_to_key:
            if word not in self.vocabulary:  # new words
                self.LRU_index.insert(0, word)
            else:  # duplicate words
                self.LRU_index.remove(word)
                self.LRU_index.insert(0, word)
        self.vocabulary = list(self.model_to_train.wv.index_to_key)

    def update_true_ref(self):
        if len(self.ref_neg) > 0:
            for words in self.ref_neg:
                if words in self.model_to_train.wv:
                    self.ref_neg.remove(words)
                    if words not in self.true_ref_neg:
                        self.true_ref_neg.append(words)
        if len(self.ref_pos) > 0:
            for words in self.ref_pos:
                if words in self.model_to_train.wv:
                    self.ref_pos.remove(words)
                    if words not in self.true_ref_pos:
                        self.true_ref_pos.append(words)

    def update_model(self, new_sentences):

        if self.flag:
            call_model = self.load_model()
            self.flag = False
        else:
            call_model = self.model_to_train

        # incremental learning
        self.incremental_training(new_sentences)
        self.update_LRU_index()
        self.update_true_ref()

    def classify_result(self, tweets):
        for t in range(len(tweets)):
            predict_result = self.predict(tweets[t], self.model_to_train)
            self.confidence_list.append(predict_result[0])

            if MODE == "LABEL":
                d = {'true_label': self.true_label[t],
                     'neg_coefficient': self.neg_coefficient,
                     'pos_coefficient': self.pos_coefficient}
                self.labelled_dataset.append([
                    self.collector[t][0], predict_result[0], predict_result[1], self.collector[t][1], d])
            self.predictions.append(predict_result[1])

        logger.info('prediction count:negative prediction = ' + str(self.predictions.count(0)) + ' positive prediction '
                                                                                                 '= ' + str(
            self.predictions.count(1)))

        self.neg_coefficient = self.predictions.count(0) / (self.predictions.count(1) + self.predictions.count(0))
        self.pos_coefficient = 1 - self.neg_coefficient
        if MODE == "LABEL":
            self.collector = []
            ans = self.labelled_dataset
        else:
            ans = accuracy_score(self.true_label, self.predictions)
        self.predictions = []
        return ans

    def predict(self, tweet, model):
        sentence = np.zeros(20)
        counter = 0
        cos_sim_bad, cos_sim_good = 0, 0
        for words in tweet:
            try:
                sentence += model.wv[words]  # np.array(list(model.wv[words]) + new_feature)
                counter += 1
            except:
                pass
        if counter != 0:
            sentence_vec = sentence / counter
        k_cur = min(len(self.true_ref_neg), len(self.true_ref_pos))
        for neg_word in self.true_ref_neg[:k_cur]:
            try:
                cos_sim_bad += dot(sentence_vec, model.wv[neg_word]) / (norm(sentence_vec) * norm(model.wv[neg_word]))
            except:
                pass
        for pos_word in self.true_ref_pos[:k_cur]:
            try:
                cos_sim_good += dot(sentence_vec, model.wv[pos_word]) / (norm(sentence_vec) * norm(model.wv[pos_word]))
            except:
                pass
        if cos_sim_bad - cos_sim_good > self.confidence:
            return cos_sim_bad - cos_sim_good, 0
        elif cos_sim_bad - cos_sim_good < -self.confidence:
            return cos_sim_good - cos_sim_bad, 1
        else:
            if cos_sim_bad * self.neg_coefficient >= cos_sim_good * self.pos_coefficient:
                return cos_sim_bad - cos_sim_good, 0
            else:
                return cos_sim_good - cos_sim_bad, 1

    def logFile(self, f, m):
        with open(f, 'a') as wr:
            wr.write(m)


def unsupervised_stream(ds, map_parallelism=1, reduce_parallelism=1):
    # ds.print()
    ds = ds.map(unsupervised_OSA()).set_parallelism(map_parallelism)
    ds = ds.filter(lambda x: x[0] != 'collecting')
    ds = ds.key_by(lambda x: x[0], key_type=Types.STRING())
    ds = ds.reduce(lambda x, y: (x[0], unsupervised_OSA().model_merge(x, y))).set_parallelism(reduce_parallelism)
    ds = ds.filter(lambda x: x[0] != 'model').map(lambda x: x[1])
    # ds = ds.map(for_output()).set_parallelism(1))
    ds = ds.flat_map(split)  # always put output_type before passing it to file sink
    # ds = ds.add_sink(StreamingFileSink  # .set_parallelism(2)
    #                  .for_row_format('./output', Encoder.simple_string_encoder())
    #                  .build())
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PLStream in two modes, labelling and accuracy. Accuracy mode is\
     default')
    # parser.add_argument('-l', dest='mode', action='store_const', default='ACC', const='LABEL',
    #                     help='Generate label(default: print accuracy)')
    args = parser.parse_args()
    MODE = 'LABEL'
    logging.basicConfig(filename='plstream.log')
    logger.info('logger initiated')

    parallelism = 4
    # the labels of dataset are only used for accuracy computation, since PLStream is unsupervised
    f = pd.read_csv('./train.csv', header=None)  # , encoding='ISO-8859-1'
    f.columns = ["label", "review"]
    # 20,000 data for quick testing
    test_N = 150000
    true_label = list(f.label)[:test_N]
    for i in range(len(true_label)):
        if true_label[i] == 1:
            true_label[i] = 0
        else:
            true_label[i] = 1

    yelp_review = list(f.review)
    yelp_review = list(f.review)[:test_N]
    print(len(yelp_review))
    data_stream = []
    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))
        # print(i, int(true_label[i]), yelp_review[i])
    print('Coming Stream is ready...')
    print('===============================')
    start_time = time()
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)
    # always update ds variable
    ds = unsupervised_stream(ds).map(lambda x: x[:-1])

    ds.print()
    env.execute("osa_job")
    print(time() - start_time)