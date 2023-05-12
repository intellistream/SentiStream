# pylint: disable=import-error
# pylint: disable=no-name-in-module

import csv
import numpy as np

from time import time, time_ns
from kafka import KafkaConsumer
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

import config

from other_exp.utils import tokenize


@dataclass
class SelfLearningConfig:
    prob_thres: int = 0.75
    pretrain_size: int = 5000
    batch_size: int = 100
    batching_flag: str = 'BATCHING'


consumer = KafkaConsumer(
    config.KAFKA_TOPIC,
    bootstrap_servers=config.BOOTSTRAP_SERVER,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: x.decode('utf-8'),
    consumer_timeout_ms=1000  # NOTE: Decrease 1 sec from total time
)


class SelfLearning():
    def __init__(self, config):
        self.config = config
        self.count_vec = CountVectorizer()
        self.model = MultinomialNB()
        self.train_node = 0
        self.train_X = []
        self.train_y = []
        self.acc = []
        self.f1 = []

    def _reset_model(self):
        self.count_vec = CountVectorizer()
        self.model = MultinomialNB()

    def _fit(self, X, y):
        vec_X = self.count_vec.fit_transform(X)
        self.model.fit(vec_X, y)

    def _predict_proba(self, X):
        vec_X = self.count_vec.transform(X)
        return self.model.predict_proba(vec_X)

    def _predict(self, X):
        vec_X = self.count_vec.transform(X)
        return self.model.predict(vec_X)

    def map(self, value):
        tag, X, y = value
        clean_X = ' '.join(tokenize(X))

        if tag == 'pretrain':
            self.train_X.append(clean_X)
            self.train_y.append(y)
            self._reset_model()
            self._fit(self.train_X, self.train_y)
        else:
            prob_y = self._predict_proba([clean_X])[0]

            label = np.argmax(prob_y)
            if prob_y[label] > self.config.prob_thres:
                self.train_X.append(clean_X)
                self.train_y.append(label)
                if len(self.train_X) - self.train_node >= self.config.batch_size:
                    self._reset_model()
                    self._fit(self.train_X, self.train_y)
                    self.train_node = len(self.train_X)

            pred_y = self._predict([clean_X])
            self.acc.append(accuracy_score([y], pred_y))
            self.f1.append(f1_score([y], pred_y, zero_division=0))

def test_sl():
    sl = SelfLearning(SelfLearningConfig(batch_size=1000))

    # train
    with open(config.TRAIN_DATA, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            sl.map(('pretrain', row[1], int(row[0])))

    start = time()
    latency = []

    # inf
    for message in consumer:
        arrival_time = time_ns()

        label, text = message.value.split('|||', 1)
        label = int(label)

        sl.map(('inf', text, label))

        latency.append(time_ns() - arrival_time)

    consumer.close()

    return (time() - start - 1, sum(latency)/(len(latency) * 1000000), sl.acc, sl.f1)
