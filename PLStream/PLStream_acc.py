import random
import copy
import re
import numpy as np

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


class for_output(MapFunction):
    def __init__(self):
        pass

    def map(self, value):
        return str(value[1])


class unsupervised_OSA(MapFunction):

    def __init__(self):
        # collection
        self.true_label = []
        self.cleaned_text = []
        self.stop_words = stopwords.words('english')
        self.collector_size = 2000

        # model pruning
        self.LRU_index = ['good', 'bad']
        self.max_index = max(self.LRU_index)
        self.LRU_cache_size = 30000
        #         self.sno = nltk.stem.SnowballStemmer('english')

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
        self.acc_to_plot = []
        self.predictions = []

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
        try:
            called_model = pickle.loads(self.redis_param.get('osamodel'))
            return called_model
        except TypeError:
            logging.info('The model name you entered cannot be found in redis')
        except (redis.exceptions.RedisError, TypeError, Exception):
            logging.warning('Unable to call the model from Redis server, please check your model')

    # tweet preprocessing
    def text_to_word_list(self, text):
        text = re.sub("@\w+ ", "", text)
        text = re.sub("[!~#$+%*:()'?-]", ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        clean_word_list = text.strip().split(' ')
        clean_word_list = [w for w in clean_word_list if w not in self.stop_words]
        while '' in clean_word_list:
            clean_word_list.remove('')
        self.cleaned_text.append(clean_word_list)
        if len(self.cleaned_text) >= self.collector_size:
            ans = self.update_model(self.cleaned_text)
            return ans
        else:
            return ('collecting', '1')

    def model_prune(self, model):

        if len(model.wv.index_to_key) <= self.LRU_cache_size:
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
        model_new.syn1 = final_syn1
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
        if model1[0] == 'acc':
            return (float(model1[1]) + float(model2[1])) / 2
        elif model1[0] == 'model':
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
                                           np.array(final_syn1neg), \
                                           final_cum_table, corpus_count, np.array(final_count),
                                           np.array(final_sample_int), \
                                           np.array(final_code), np.array(final_point), model1)
            self.save_model(model_new)
            self.flag = True
            return model_new

    def map(self, tweet):

        self.true_label.append(int(tweet[1]))
        return self.text_to_word_list(tweet[0])

    def update_model(self, new_sentences):

        if self.flag:
            call_model = self.load_model()
            self.flag = False
        else:
            call_model = self.model_to_train

        # incremental learning
        call_model.build_vocab(new_sentences, update=True)  # 1) update vocabulary
        call_model.train(new_sentences,  # 2) incremental training
                         total_examples=call_model.corpus_count,
                         epochs=call_model.epochs)
        for word in call_model.wv.index_to_key:
            if word not in self.vocabulary:  # new words
                self.LRU_index.insert(0, word)
            else:  # duplicate words
                self.LRU_index.remove(word)
                self.LRU_index.insert(0, word)
        self.vocabulary = list(call_model.wv.index_to_key)
        self.model_to_train = call_model

        if len(self.ref_neg) > 0:
            for words in self.ref_neg:
                if words in call_model.wv:
                    self.ref_neg.remove(words)
                    if words not in self.true_ref_neg:
                        self.true_ref_neg.append(words)
        if len(self.ref_pos) > 0:
            for words in self.ref_pos:
                if words in call_model.wv:
                    self.ref_pos.remove(words)
                    if words not in self.true_ref_pos:
                        self.true_ref_pos.append(words)

        classify_result = self.eval(new_sentences, call_model)
        self.cleaned_text = []
        self.true_label = []

        if time() - self.timer >= self.time_to_reset:
            call_model = self.model_prune(call_model)
            model_to_merge = ('model', call_model)
            self.timer = time()
            return model_to_merge
        else:
            not_yet = ('acc', classify_result)
            return not_yet

    def eval(self, tweets, model):
        for tweet in tweets:
            self.predictions.append(self.predict(tweet, model))

        self.neg_coefficient = self.predictions.count(0) / len(tweets)
        self.pos_coefficient = 1 - self.neg_coefficient
        ans = accuracy_score(self.true_label, self.predictions)
        self.predictions = []

        return str(ans)

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
            return 0
        elif cos_sim_bad - cos_sim_good < self.confidence * -1:
            return 1
        else:
            if cos_sim_bad * self.neg_coefficient >= cos_sim_good * self.pos_coefficient:
                return 0
            else:
                return 1


if __name__ == '__main__':
    from pyflink.datastream.connectors import StreamingFileSink
    from pyflink.common.serialization import Encoder
    from time import time
    import pandas as pd

    parallelism = 4
    
    # the labels of dataset are only used for accuracy computation, since PLStream is unsupervised
    f = pd.read_csv('./train.csv', names=['label', 'review'])
    # , encoding='ISO-8859-1'
    
    # 80,000 data for quick testing
    df = f[:80000]
    df['label'] -= 1

    true_label = df['label'].tolist()
    yelp_review = df['review'].tolist()

    data_stream = []
    for i in range(len(yelp_review)):
        data_stream.append((yelp_review[i], int(true_label[i])))

    print('Coming Stream is ready...')
    print('===============================')

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    ds = env.from_collection(collection=data_stream)
    ds.map(unsupervised_OSA()).set_parallelism(parallelism) \
        .filter(lambda x: x[0] != 'collecting') \
        .key_by(lambda x: x[0], key_type=Types.STRING()) \
        .reduce(lambda x, y: (x[0], unsupervised_OSA().model_merge(x, y))).set_parallelism(2) \
        .filter(lambda x: x[0] != 'model') \
        .map(for_output(), output_type=Types.STRING()).set_parallelism(1) \
        .add_sink(StreamingFileSink  # .set_parallelism(2)
                  .for_row_format('./output', Encoder.simple_string_encoder())
                  .build())
    env.execute("osa_job")
