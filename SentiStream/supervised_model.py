# import redis

from pyflink.datastream.functions import RuntimeContext, MapFunction

# from ann_model import Model
from han_model import Model
from utils import load_data, default_model_pretrain, train_word2vec, generate_vector_mean, process_batch

pseudo_data_folder = './senti_output'

class ModelTrain(MapFunction):
    """
    Class for training classifier.
    """

    def __init__(self, pseudo_data_collection_threshold, accuracy_threshold):
        """Initialize class

        Parameters:
            train_data_size (int): size of training data.
        """
        self.model = None
        self.pseudo_data_collection_threshold = pseudo_data_collection_threshold
        self.accuracy_threshold = accuracy_threshold
        self.labels = []
        self.sentences = []

        # self.redis = None

    def open(self, runtime_context: RuntimeContext):
        """Initialize word vector model before starting stream/batch processing.

        Parameters:
            runtime_context (RuntimeContext): give access to Flink runtime env.
        """
        self.model = default_model_pretrain("ssl-w2v.model")

        train_df = load_data(pseudo_data_folder)
        
        self.labels = train_df.label
        self.sentences = train_df.review
        self.dummy_sent = train_df.review

        # self.redis = redis.StrictRedis(host='localhost', port=6379, db=0)

    def train_classifier(self, mean_vectors):
        """Intialize and train sentiment classifier on train data

        Parameters:
            mean_vectors (list): list of average word vectors for each sentences.

        Returns:
            T: trained sentiment classifier
        """

        # clf = Model(mean_vectors, self.labels,
        #             self.model.vector_size, False)
        clf = Model(self.dummy_sent, self.labels, self.model.wv.key_to_index, [self.model.wv[key].tolist() for key in self.model.wv.index_to_key], False)
        clf.fit_and_save('ssl-clf.pth')

        # TODO CONTINOUS TRAIN

    def map(self, acc):
        """Map function to collect train data for classifier model and train it.

        Parameters:
            tweet (tuple): tuple of tweet and it's label

        Returns:
            (str): 'fininshed training' if model is trained, else, if collecting data for training, 
            'collecting'
        """


        if (len(self.labels) > self.pseudo_data_collection_threshold and acc < self.accuracy_threshold):
            self.sentences = process_batch(self.sentences)

            self.train_wordvector_model()

            mean_vectors = []
            for sentence in self.sentences:
                mean_vectors.append(generate_vector_mean(self.model, sentence))

            self.train_classifier(mean_vectors)
            # try:
            #     self.redis.set('word_vector_update', int(True))
            #     self.redis.set('classifier_update', int(True))
            # except ConnectionError:
            #     raise ConnectionError('Failed to open redis')

            return "finished training"
            
        else:
            return (f'acc: {acc}, threshold: {self.accuracy_threshold}\npseudo_data_size: {len(self.labels)} threshold: {self.pseudo_data_collection_threshold}')


    def train_wordvector_model(self, func=train_word2vec):
        """Train word vector model

        Parameters:
            func (function, optional): function to train model. Defaults to train_word2vec.
        """
        func(self.model, self.sentences, 'ssl-w2v.model')

def supervised_model(ds, pseudo_data_collection_threshold=0.0, accuracy_threshold=0.0):
    ds.map(ModelTrain(pseudo_data_collection_threshold, accuracy_threshold)).print()

if __name__ == '__main__':
    # data source
    pseudo_data_folder = './senti_output'
    train_data_file = 'exp_train.csv'

    # config.PSEUDO_DATA_COLLECTION_THRESHOLD = 0
    # config.ACCURACY_THRESHOLD = 0.9

    parallelism = 1

    # data sets
    pseudo_data_size, train_df = load_data(pseudo_data_folder, train_data_file)

    # redis_param = redis.StrictRedis(host='localhost', port=6379, db=0)
    # accuracy = float(redis_param.get('batch_inference_accuracy').decode())
    accuracy = 0.4
    supervised_model(parallelism, train_df, pseudo_data_size, accuracy)

