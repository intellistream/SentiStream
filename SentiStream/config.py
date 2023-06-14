# Kafka
KAFKA_TOPIC = 'sentiment-data'
BOOTSTRAP_SERVER = 'localhost:9092'  # default 9092

# Input files
DATA = 'data/data_0_5_percent.csv'
TRAIN_DATA = 'data/data_train_0_5_percent.csv'

# Outputs
FINISHED = 'MODEL_TRAINED'
SKIPPED = 'SKIPPED_TRAINING'

# Model filenames
SSL_WV = 'ssl-wv.model'
SSL_CLF = 'ssl-clf.pth'
