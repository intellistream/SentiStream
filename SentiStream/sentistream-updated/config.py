# pylint: disable=import-error
from gensim.models import Word2Vec, FastText

# PyFlink
PYFLINK = False

# SentiStream
SSL_MODEL = 'ANN'  # 'HAN', 'ANN'
# Word2Vec, FastText ---------------WORD2VEC is BETTER IN PERFORMANCE
WORD_VEC_ALGO = Word2Vec
STEM = True  # TODO: REMOVE FROM HERE

# Kafka
KAFKA_TOPIC = 'sentiment-data'
BOOTSTRAP_SERVER = 'localhost:9092'

# Input files
DATA = 'train.csv'

# Outputs
BATCHING = 'BATCHING'
LOW_CONF = 'LOW_CONFIDENCE'
FINISHED = 'MODEL_TRAINED'
SKIPPED = 'SKIPPED_TRAINING'

# Model filenames
US_WV = 'plstream-wv.model'
SSL_WV = 'ssl-wv.model'
SSL_CLF = 'ssl-clf.pth'
