import sys
import redis
import logging
import random
import numpy as np

from sklearn.metrics import accuracy_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode

from utils import load_data, process_text_and_generate_tokens, generate_vector_mean, default_model_pretrain

# logger
logger = logging.getLogger('SentiStream')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('sentistream.log', mode='w')
formatter = logging.Formatter('SentiStream:%(thread)d %(lineno)d: %(levelname)s: %(asctime)s %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p', )
fh.setFormatter(formatter)
logger.addHandler(fh)

# supress warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Preprocessor(MapFunction):
    """
    Class for data preprocessing
    """

    def __init__(self, test_data_size, parallelism):
        """Initialize class

        Parameters:
            test_data_size (int_): size of testing data
            parallelism (int): number of parallelism
        """
        self.model = None
        self.collector = []
        self.output = []
        self.collector_size = int(test_data_size / parallelism)

    def open(self, runtime_context: RuntimeContext):
        """Initialize word embedding model before starting stream/batch processing

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = default_model_pretrain(
            'PLS_c10.model')

    def map(self, tweet):
        """Map function to collect and preprocess data for classifier model.

        Parameters:
            tweet (tuple): tuple of tweet and it's label

        Returns:
            (str or list): list of label and avg word vector of tweet if all data per processing
            unit is collected else, 'collecting'
        """
        processed_text = process_text_and_generate_tokens(tweet[1])
        vector_mean = generate_vector_mean(self.model, processed_text)
        self.collector.append([tweet[0], vector_mean])

        # if len(self.collector) >= self.collector_size:
        #     self.output += self.collector
        #     # WHY DO WE NEED THIS STEP ?????????????????????????????????
        #     self.collector = []
        #     return self.output
        # else:
        #     return 'collecting'

        # NOTE: Collector size = total test data size so no need to have self.output????? (even if
        # in parallel env, collector size will be size of data in single process) so temp
        # implementation,,, # as we are not reusing same class obj it won't add up
        if len(self.collector) >= self.collector_size:
            return self.collector
        else:
            return 'collecting'


class Predictor(MapFunction):
    """
    Class for Predictor(Classifier) Placeholder
    """

    def __init__(self):
        """Initialize class"""
        self.model = None
        self.labels = []
        self.data = []

    # def open(self, runtime_context: RuntimeContext):
    #     """Initialize classifier model before starting stream/batch processing

    #     Parameters:
    #         runtime_context (RuntimeContext): give access to Flink runtime env.
    #     """
    #     self.model = default_model_classifier()  # change to your model

    def get_accuracy(self, predictions, func=accuracy_score):
        """Calculate accuracy of model's prediction

        Parameters:
            predictions (list): list of predicted labels
            func (function): function to calculate accuracy

        Returns:
            (float) accuracy of predicted labels
        """
        return func(self.labels, predictions)

    def get_prediction(self, func):
        """Predict sentiment of text

        Parameters:
            func (function): prediction function of model

        Returns:
            (float) predicted label
        """
        return func(self.data)

    #TODO: Remove
    def get_dummy_prediction(self):
        """Generate dummy (random) prediction

        Returns:
            (float): predicted label
        """
        return [random.choice([0, 1]) for _ in range(len(self.data))]

    def map(self, ls):
        """Map function to predict label on batch data.

        Parameters:
            ls (list): list of text and it's labels

        Returns:
            (tuple): (1, accuracy of model's prediction)
        """
        self.labels, self.data = zip(*ls)

        # change to your prediction function
        # predictions = self.get_prediction(self.model.predict)
        predictions = self.get_dummy_prediction()
        # change to your accuracy function
        accuracy = self.get_accuracy(predictions)
        return 1, accuracy


def batch_inference(ds, test_data_size, preprocess_parallelism=1, classifier_parallelism=1):
    """Predict label of incoming batch data

    Args:
        ds (datastream): stream of data from source
        test_data_size (int): size of data
        preprocess_parallelism (int, optional): number of parallelism for preprocessing. 
        Defaults to 1.
        classifier_parallelism (int, optional): number of parallelism for prediction. Defaults to 1.

    Returns:
        _type_: _description_
    """
    redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    ds = ds.map(Preprocessor(test_data_size, preprocess_parallelism)) \
        .set_parallelism(preprocess_parallelism) \
        .filter(lambda i: i != 'collecting')

    ds = ds.map(Predictor()).set_parallelism(classifier_parallelism) \
        .key_by(lambda x: x[0]) \
        .reduce(lambda x, y: (1, (x[1] + y[1]) / 2))

    with ds.execute_and_collect() as results:
        for accuracy in results:
            redis_param.set('batch_inference_accuracy', accuracy[1].item())
    return accuracy[1]


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO, format="%(message)s")

    pseudo_data_folder = './senti_output'
    test_data_file = './exp_test.csv'

    pseudo_data_size, test_df = load_data(pseudo_data_folder, test_data_file)
    test_data_size = len(test_df)

    true_label = test_df.label
    yelp_review = test_df.review

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
    accuracy = batch_inference(ds, test_data_size)

    print(accuracy)

# STREAM DATA TO BATCH DATA ??????
# NOTE: loading batch data, so no need to wait & collect ---

# NOTE: Why appending 1 infront of each tuple?
