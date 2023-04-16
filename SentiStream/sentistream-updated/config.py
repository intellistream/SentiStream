# pylint: disable=import-error
from gensim.models import Word2Vec, FastText

# PyFlink
PYFLINK = True

# SentiStream
SSL_MODEL = 'HAN'  # 'HAN', 'ANN'
WORD_VEC_ALGO = Word2Vec  # Word2Vec, FastText
STEM = True  # TODO: REMOVE FROM HERE

# Kafka
KAFKA_TOPIC = 'sentiment-data'
BOOTSTRAP_SERVER = 'localhost:9092'

# Input files
DATA = 'train.csv'

# Outputs
BATCHING = 'BATCHING'
LOW_CONF = 'LOW_CONFIDENCE'

# Model filenames
US_WV = 'plstream-wv.model'
SSL_WV = 'ssl-wv.model'
SSL_CLF = 'ssl-clf.pth'
