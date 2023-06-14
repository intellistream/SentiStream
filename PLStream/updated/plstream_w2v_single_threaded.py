import nltk
import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy import float32 as REAL
from pandas import read_csv

from gensim.models import Word2Vec

from sklearn.metrics import accuracy_score, f1_score

from utils import cos_similarity, clean_sentence, make_input_collection

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.common.typeinfo import Types

logger = logging.getLogger('PLStream')
logger.setLevel(logging.DEBUG)

LABEL_NEG, LABEL_POS = 0, 1
BATCHING_FLAG = 'BATCHING'
BATCH_SIZE = 250
VECTOR_SIZE = 20

CONFIDENCE = 0.5
TTD = True

REF_POS = [
    'love', 'best', 'beautiful', 'great',
            'cool', 'awesome', 'wonderful',
            'brilliant', 'excellent', 'fantastic']  # CHECK OUTPUT OF STEMMER??????????????
REF_NEG = [
    'bad', 'worst', 'stupid', 'disappointing',
    'terrible', 'rubbish', 'boring', 'awful',
    'unwatchable', 'awkward']

class PLStream(MapFunction):
    def __init__(self):
        self.neg_coef = 0.5
        self.pos_coef = 0.5
        self.neg_count = 0
        self.pos_count = 0
        self.update_model = False
        super().__init__()

    def open(self, runtime_context: RuntimeContext):
        self.model = Word2Vec(vector_size=VECTOR_SIZE, workers=12)
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.batch_X, self.batch_y = [], []

    def map(self, value):
        tag, _, X, y = value
        clean_X = clean_sentence(X, self.stemmer)

        batch_size, acc, f1 = 1, 0, 0
        if tag == 'train':
            self.batch_X.append(clean_X)
            self.batch_y.append(y)
            if len(self.batch_X) >= BATCH_SIZE:
                self.model.build_vocab(self.batch_X, update=self.update_model) # REMOVE LRU
                self.model.train(self.batch_X, total_examples=self.model.corpus_count, epochs=self.model.epochs)
                acc, f1 = self._eval_model(
                    self.batch_X, self.batch_y, is_train=True)
                batch_size = len(self.batch_X)
                self.batch_X, self.batch_y = [], []
                self.update_model = True
            else:
                tag = BATCHING_FLAG
        elif tag == 'eval':
            self.batch_X.append(clean_X)
            self.batch_y.append(y)
            if len(self.batch_X) >= BATCH_SIZE:
                acc, f1 = self._eval_model(
                    self.batch_X, self.batch_y, is_train=False)
                batch_size = len(self.batch_X)
                self.batch_X, self.batch_y = [], []
            else:
                tag = BATCHING_FLAG

        return (tag, batch_size, acc, f1)

    def _predict(self, sentence):
        vector = np.zeros(VECTOR_SIZE, dtype=REAL)
        counter = 0
        for word in sentence:
            if word in self.model.wv.key_to_index:
                vector += self.model.wv[word]
                counter += 1
        if counter != 0:
            vector = vector / counter

        cos_sim_neg = 0
        for word in REF_NEG:
            if word in self.model.wv.key_to_index:
                cos_sim_neg += cos_similarity(vector, self.model.wv[word])

        cos_sim_pos = 0
        for word in REF_POS:
            if word in self.model.wv.key_to_index:
                cos_sim_pos += cos_similarity(vector, self.model.wv[word])

        if cos_sim_neg - cos_sim_pos > CONFIDENCE:
            return cos_sim_neg - cos_sim_pos, LABEL_NEG
        elif cos_sim_pos - cos_sim_neg > CONFIDENCE:
            return cos_sim_pos - cos_sim_neg, LABEL_POS
        elif TTD:
            if cos_sim_neg * self.neg_coef >= cos_sim_pos * self.pos_coef:
                return cos_sim_neg - cos_sim_pos, LABEL_NEG
            else:
                return cos_sim_pos - cos_sim_neg, LABEL_POS
        else:
            if cos_sim_neg > cos_sim_pos:
                return cos_sim_neg - cos_sim_pos, LABEL_NEG
            else:
                return cos_sim_pos - cos_sim_neg, LABEL_POS

    def _eval_model(self, batch_X, batch_y, is_train):
        confidences, pred_y = [], []
        for sentence in batch_X:
            confi, y = self._predict(sentence)
            confidences.append(confi)
            pred_y.append(y)

        if is_train:
            self.neg_count += pred_y.count(1)
            self.pos_count += pred_y.count(0)
            total = self.neg_count + self.pos_count
            self.neg_coef = self.neg_count / total
            self.pos_coef = self.pos_count / total

        acc = accuracy_score(batch_y, pred_y)
        f1 = f1_score(batch_y, pred_y)

        return acc, f1


def run(collection, parallelism=1):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)

    plstream = env.from_collection(collection).name('CollectionSource') \
        .map(PLStream()).name('PLStream').set_parallelism(parallelism) \
        .filter(lambda x: x[0] != 'BATCHING') \
        .map(lambda x: f'{x[1]} - {x[2]} - {x[3]}', output_type=Types.STRING()).set_parallelism(1)

    acc = []

    with plstream.execute_and_collect() as results:
        for temp in results:
            acc.append(float(temp.split(' - ')[1]))

    plt.figure(figsize=(10, 5))

    plt.plot([x*(BATCH_SIZE) for x in range(len(acc))], acc)
    plt.savefig(f'gensim-w2v.png', bbox_inches='tight')


if __name__ == '__main__':
    collection = None

    data_size = 1000
    yelp = read_csv("train.csv")
    yelp.columns = ['label', 'sentence']
    yelp_train_X, yelp_train_y = list(yelp.sentence), list(yelp.label-1)

    collection = []
    collection = make_input_collection(
        'train', yelp_train_X[:data_size], yelp_train_y[:data_size])
    
    run(collection)
