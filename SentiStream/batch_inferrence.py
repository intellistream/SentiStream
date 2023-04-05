import sys
# import redis
import torch

from sklearn.metrics import accuracy_score
from pyflink.datastream.functions import RuntimeContext, MapFunction
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream import CheckpointingMode

from utils import load_data, process_text_and_generate_tokens, generate_vector_mean, default_model_pretrain, load_torch_model


class Preprocessor(MapFunction):
    """
    Class for data preprocessing
    """

    def __init__(self,  parallelism):
        """Initialize class

        Parameters:
            test_data_size (int_): size of testing data
            parallelism (int): number of parallelism
        """
        self.model = None
        self.collector = []
        # self.collector_size = int(test_data_size / parallelism)
        self.collector_size = 16

    def open(self, runtime_context: RuntimeContext):
        """Initialize word embedding model before starting stream/batch processing

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = default_model_pretrain('w2v.model')

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
            output = self.collector
            self.collector = []
            return output
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

    def open(self, runtime_context: RuntimeContext):
        """Initialize classifier model before starting stream/batch processing

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = load_torch_model('model.pth')

    def get_accuracy(self, predictions, func=accuracy_score):
        """Calculate accuracy of model's prediction

        Parameters:
            predictions (list): list of predicted labels
            func (function): function to calculate accuracy

        Returns:
            (float) accuracy of predicted labels
        """
        return func(self.labels, predictions)

    def get_prediction(self):
        """Predict sentiment of text

        Returns:
            (float) predicted label
        """
        return self.model(torch.tensor(self.data, dtype=torch.float32))

    def map(self, ls):
        """Map function to predict label on batch data.

        Parameters:
            ls (list): list of text and it's labels

        Returns:
            (tuple): (1, accuracy of model's prediction)
        """
        self.labels, self.data = zip(*ls)

        with torch.no_grad():
            predictions = self.get_prediction()

        accuracy = self.get_accuracy(torch.round(predictions))
        return 1, accuracy


def batch_inference(ds, preprocess_parallelism=1, classifier_parallelism=1):
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
    # redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    ds = ds.map(Preprocessor(preprocess_parallelism)) \
        .set_parallelism(preprocess_parallelism) \
        .filter(lambda i: i != 'collecting')

    ds = ds.map(Predictor()).set_parallelism(classifier_parallelism) \
        .key_by(lambda x: x[0]) \
        .reduce(lambda x, y: (1, (x[1] + y[1]) / 2)) \
        .map(lambda x: x[1])
    
    return ds


if __name__ == '__main__':
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