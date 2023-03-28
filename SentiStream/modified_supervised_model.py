import sys
import redis
import logging

import config
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode
from pyflink.datastream.functions import RuntimeContext, MapFunction

from ann_model import Model
from modified_batch_inferrence import batch_inference
from utils import load_data, pre_process, default_model_pretrain, train_word2vec, generate_vector_mean


class ModelTrain(MapFunction):
    """
    Class for training classifier.
    """

    def __init__(self, train_data_size, init):
        """Initialize class

        Parameters:
            train_data_size (int): size of training data.
        """
        self.model = None
        self.init = init
        self.sentences = []
        self.labels = []
        self.output = []
        self.collection_size = train_data_size
        self.redis = None

    def open(self, runtime_context: RuntimeContext):
        """Initialize word vector model before starting stream/batch processing.

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = default_model_pretrain(
            "PLS_c10.model" if self.init else "w2v.model")
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def train_classifier(self, mean_vectors):
        """Intialize and train sentiment classifier on train data

        Parameters:
            mean_vectors (list): list of average word vectors for each sentences.

        Returns:
            T: trained sentiment classifier
        """

        clf = Model(mean_vectors, self.labels, self.model.vector_size, self.init)
        clf.fit_and_save('model.pth')

        # TODO CONTINOUS TRAIN

    def map(self, tweet):
        """Map function to collect train data for classifier model and train it.

        Parameters:
            tweet (tuple): tuple of tweet and it's label

        Returns:
            (str): 'fininshed training' if model is trained, else, if collecting data for training, 
            'collecting'
        """
        self.sentences.append(tweet[1])
        self.labels.append(tweet[0])
        if len(self.labels) >= self.collection_size:
            self.train_wordvector_model()

            mean_vectors = []
            for sentence in self.sentences:
                mean_vectors.append(generate_vector_mean(self.model, sentence))

            self.train_classifier(mean_vectors)
            try:
                self.redis.set('word_vector_update', int(True))
                self.redis.set('classifier_update', int(True))
            except ConnectionError:
                raise ConnectionError('Failed to open redis')

            return "finished training"
        else:
            return 'collecting'

    def train_wordvector_model(self, func=train_word2vec):
        """Train word vector model

        Parameters:
            func (function, optional): function to train model. Defaults to train_word2vec.
        """
        func(self.model, self.sentences, 'w2v.model')


def supervised_model(data_process_parallelism, train_df, pseudo_data_size, accuracy, init=False):
    """Train supervised model and word vector model

    Parameters:
        data_process_parallelism (int): no of parallelism for training.
        train_df (DataFrame): train data.
        pseudo_data_size (int): size of pseudo data.
        accuracy (float): accuracy of supervised model before training.
        init (bool, optional): whether training for first time or retraining. Defaults to False.
    """
    if init or (pseudo_data_size > config.PSEUDO_DATA_COLLECTION_THRESHOLD and accuracy < config.ACCURACY_THRESHOLD):

        # data preparation
        true_label = train_df.label
        yelp_review = train_df.review

        data_stream = []

        for i in range(len(yelp_review)):
            data_stream.append((int(true_label[i]), yelp_review[i]))

        # stream environment setup
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_runtime_mode(RuntimeExecutionMode.BATCH)
        env.set_parallelism(1)
        env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

        print('Coming Stream is ready...')
        print('===============================')

        # data stream pipline
        ds = env.from_collection(collection=data_stream)
        ds = ds.map(pre_process)  # change to your pre_processing function,
        ds = ds.set_parallelism(data_process_parallelism).map(
            ModelTrain(len(train_df), init))
        ds = ds.filter(lambda x: x != 'collecting')
        # ds = batch_inference(env.from_collection(
        #     collection=data_stream), len(train_df))

        ds.print()

        env.execute()
    else:
        print("accuracy below threshold: " +
              str(accuracy < config.ACCURACY_THRESHOLD))
        print("pseudo data above threshold: " +
              str(pseudo_data_size > config.PSEUDO_DATA_COLLECTION_THRESHOLD))
        print("Too soon to update model")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO, format="%(message)s")
    # data source
    pseudo_data_folder = './senti_output'
    train_data_file = 'exp_train.csv'

    config.PSEUDO_DATA_COLLECTION_THRESHOLD = 0
    config.ACCURACY_THRESHOLD = 0.9

    parallelism = 1

    # data sets
    pseudo_data_size, train_df = load_data(pseudo_data_folder, train_data_file)

    redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    # accuracy = float(redis_param.get('batch_inference_accuracy').decode())
    accuracy = 0.4
    supervised_model(parallelism, train_df, pseudo_data_size, accuracy)

# TODO: WHY WAITING IN BATCH MODE ????
