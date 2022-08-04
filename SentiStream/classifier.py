import logging
import sys

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext, MapFunction
import time
from utils import process_text_and_generate_tokens, split, generate_vector_mean, default_model_pretrain, \
    default_model_classifier
import pandas as pd
from pyflink.datastream import CheckpointingMode
import redis
from time import time


def join(x, y):
    # logging.warning('x is: ' + str(x))
    # logging.warning('y is: ' + str(y))
    return x + y[1]


class Supervised_OSA_inference(MapFunction):
    def __init__(self, with_accuracy=True):
        """
        :param with_accuracy: True if labels are provided to the datastream. default value is True
        """
        self.model = None
        self.data = []
        self.collector = []
        self.output = []
        self.collector_size = 5
        self.with_accuracy = with_accuracy
        self.redis = None  # do not set redis variable here it gives error
        self.time_threshold = 10 * 60  # time threshold set to 10 mins
        self.start_time = time()

    def open(self, runtime_context: RuntimeContext):
        """
        upload model here in runtime
        """
        self.model = default_model_pretrain()  # change to your model here
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def load(self):
        """
        upload model periodically
        """
        #  periodic load
        if time() - self.start_time >= self.time_threshold:
            self.start_time = 0
            try:
                flag = int(self.redis.get('word_vector_update'))
                if flag == 1:
                    self.model = default_model_pretrain()  # change to your model here
                    self.redis.set('word_vector_update', int(False))
                elif flag is None:
                    raise RuntimeError('word_vector model update flag not set. Train model before running')
            except:
                raise ConnectionError('Failed to open redis')

    def map(self, tweet):
        # logging.warning(tweet)
        self.data.append(tweet)

        if self.with_accuracy:
            content = tweet[2]

        processed_text = process_text_and_generate_tokens(content)
        # logging.warning(processed_text)
        vector_mean = generate_vector_mean(self.model, processed_text)
        self.collector.append([tweet[0], vector_mean, content])

        if self.with_accuracy:
            self.collector[-1].append(tweet[1])

        if len(self.collector) >= self.collector_size:
            for e in self.collector:
                self.output.append(e)
            self.collector = []
            return self.output
        else:
            return 'collecting'


def default_confidence(model, data):
    return model.predict_proba(data)


class Classifier(MapFunction):
    def __init__(self):
        self.model = None
        self.data = []
        self.redis = None  # setting redis-param here gives errors
        self.time_threshold = 10 * 60  # time threshold set to 10 mins
        self.start_time = time()

    def open(self, runtime_context: RuntimeContext):
        """
        upload model here  in runtime
        """
        self.model = default_model_classifier()  # change to your model
        self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def load(self):
        """
        upload model periodically
        """
        #  periodic load
        if time() - self.start_time >= self.time_threshold:
            self.start_time = 0
            try:
                flag = int(self.redis.get('classifier_update'))
                if flag == 1:
                    self.model = default_model_classifier()  # change to your model here
                    self.redis.set('classifier_update', int(False))
                elif flag is None:
                    raise RuntimeError('classifier_model update flag not set. Train model before running')
            except:
                raise ConnectionError('Failed to open redis')

    def get_confidence(self, func=default_confidence):
        """
        :param func:a function(classifier,[vector_mean]) that is expects a classifier
                model and data of mean vectors as input
        :return: an array of confidence for 0 and 1 labels e.g. [0.40,0.60]
        """
        return func(self.model, self.data)

    def map(self, ls):
        for i in range(len(ls)):
            self.data.append(ls[i][1])
        confidence = self.get_confidence()
        for i in range(len(ls)):
            if confidence[i][0] >= confidence[i][1]:
                ls[i][1] = confidence[i][0]
                ls[i].insert(2, 0)
            else:
                ls[i][1] = confidence[i][1]
                ls[i].insert(2, 1)

        return ls


def classifier(ds):
    ds = ds.map(Supervised_OSA_inference()).filter(lambda i: i != 'collecting')
    # ds.print()
    # ds.flat_map(split).print() #data size is uneven due to use of collector
    ds = ds.map(Classifier()).flat_map(split)
    return ds


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    start_time = time()

    # input_path = './yelp_review_polarity_csv/test.csv'
    # if input_path is not None:
    f = pd.read_csv('./exp_train.csv', header=None)  # , encoding='ISO-8859-1'
    f.columns = ["label", "review"]

    f.loc[f['label'] == 1, 'label'] = 0
    f.loc[f['label'] == 2, 'label'] = 1

    true_label = f.label
    yelp_review = f.review
    data_stream = []

    for i in range(len(yelp_review)):
        data_stream.append((i, int(true_label[i]), yelp_review[i]))
        # print((i, int(true_label[i]), yelp_review[i]))

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    ds = env.from_collection(collection=data_stream)

    #  always specify output_type when writing to file

    ds = classifier(ds)

    # .key_by(lambda x: x[0])
    ds.print()
    env.execute()
    # supervised_learning(known_args.input, known_args.output)
    logging.info("time taken for execution is: " + str(time() - start_time))
