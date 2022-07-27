import pickle
import sys

import redis
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.functions import RuntimeContext, MapFunction
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import load_and_augment_data, pre_process
from gensim.models import Word2Vec

# global variables
PSEUDO_DATA_COLLECTION_THRESHOLD = 0
ACCURACY_THRESHOLD = 0.9
parallelism = 1
train_data_size = 0


class Supervised_OSA(MapFunction):
    def __init__(self, train_data_size):
        self.model = None
        self.sentences = []
        self.labels = []
        self.output = []
        self.collection_threshold = train_data_size

        # logging.warning("pseudo_data_size: " + str(pseudo_data_size))

    def open(self, runtime_context: RuntimeContext):
        self.model = Word2Vec.load('word2vec20tokenised.model')

    def map(self, tweet):
        # logging.warning(tweet)

        self.sentences.append(tweet[1])
        self.labels.append(tweet[0])
        if len(self.labels) >= self.collection_threshold:
            self.train_word2vec()

            model_vector = [(np.mean([self.model.wv[token] for token in row], axis=0)).tolist() for row in
                            self.sentences]

            clf_word2vec = RandomForestClassifier()
            clf_word2vec.fit(model_vector, self.labels)

            filename = 'supervised.model'
            pickle.dump(clf_word2vec, open(filename, 'wb'))

            return "finished training"
        else:
            return 'collecting'

    def train_word2vec(self):
        self.model.build_vocab(self.sentences, update=True)
        self.model.train(self.sentences,
                         total_examples=self.model.corpus_count,
                         epochs=self.model.epochs)


def supervised_model(data_process_parallelism, train_df, train_data_size, pseudo_data_size,
                     PSEUDO_DATA_COLLECTION_THRESHOLD,
                     accuracy,
                     ACCURACY_THRESHOLD):
    if pseudo_data_size > PSEUDO_DATA_COLLECTION_THRESHOLD and accuracy < ACCURACY_THRESHOLD:
        true_label = train_df.label
        yelp_review = train_df.review
        data_stream = []
        for i in range(len(yelp_review)):
            data_stream.append((int(true_label[i]), yelp_review[i]))

        print('Coming Stream is ready...')
        print('===============================')

        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_runtime_mode(RuntimeExecutionMode.BATCH)
        env.set_parallelism(1)
        env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
        ds = env.from_collection(collection=data_stream)

        ds = ds.map(pre_process).set_parallelism(data_process_parallelism).map(Supervised_OSA(train_data_size)) \
            .filter(lambda x: x != 'collecting')
        # ds = batch_inference(ds)
        ds.print()
        env.execute()
    else:
        print("accuracy below threshold: " + str(accuracy < ACCURACY_THRESHOLD))
        print("pseudo data above threshold: " + str(pseudo_data_size > PSEUDO_DATA_COLLECTION_THRESHOLD))
        print("Too soon to update model")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # data source
    pseudo_data_folder = './senti_output'
    train_data_file = './exp_train.csv'

    # data sets
    pseudo_data_size, train_df = load_and_augment_data(pseudo_data_folder, train_data_file)

    train_data_size = len(train_df)

    redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    accuracy = float(redis_param.get('batch_inference_accuracy').decode())
    supervised_model(parallelism, train_df, train_data_size, pseudo_data_size, PSEUDO_DATA_COLLECTION_THRESHOLD,
                     accuracy,
                     ACCURACY_THRESHOLD)
