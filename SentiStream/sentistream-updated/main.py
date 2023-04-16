# pylint: disable=import-error
# pylint: disable=no-name-in-module


# STEM OR NOT?

# TORCH JIT ?
# TODO: FIX CLASS IMBALANCE IN INITIAL TRAINING - DONE - REWATCH

from itertools import zip_longest
from time import time

import pandas as pd

from kafka import KafkaConsumer

from gensim.models import Word2Vec, FastText

from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer


from train.supervised import TrainModel
from train.pseudo_labeler import SentimentPseudoLabeler, PseudoLabelerCoMap
from unsupervised_models.plstream import PLStream
from inference.classifier import Classifier

from utils import tokenize

PYFLINK = False
FLINK_PARALLELISM = 1

SSL_MODEL = 'HAN'  # 'HAN', 'ANN'
WORD_VEC_ALGO = Word2Vec  # Word2Vec, FastText

KAFKA_TOPIC = 'sentiment-data'

STEM = True

start = time()

# ---------------- Initial training of classifier ----------------
# Train word vector {Word2Vec or FastText} on {nrows=1000} data and get word embeddings, then use
# that embedding to train NN sentiment classifier {ANN or HAN}
TrainModel(word_vector_algo=WORD_VEC_ALGO,
           ssl_model=SSL_MODEL, init=True, nrows=1000)


# ---------------- Generate pesudo labels ----------------

df = pd.read_csv('train.csv', names=[
    'label', 'review'], nrows=1000)
df['label'] -= 1

plstream = PLStream(word_vector_algo=WORD_VEC_ALGO, is_stem=STEM)
classifier = Classifier(word_vector_algo=WORD_VEC_ALGO, ssl_model=SSL_MODEL)
pseduo_labeler = SentimentPseudoLabeler()
inference = Classifier(word_vector_algo=WORD_VEC_ALGO,
                       ssl_model=SSL_MODEL, is_eval=True)

if PYFLINK:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(FLINK_PARALLELISM)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    data_stream = [(idx, label, text)
                   for idx, (label, text) in enumerate(df.values)]

    ds = env.from_collection(collection=data_stream)

    ds = ds.map(lambda x: (x[0], x[1], tokenize(x[2], STEM)))

    ds_us = ds.map(plstream.process_data).filter(
        lambda x: x != 'BATCHING').flat_map(lambda x: x)
    ds_ss = ds.map(classifier.classify).filter(
        lambda x: x != 'BATCHING').flat_map(lambda x: x)

    ds = ds_us.connect(ds_ss).map(PseudoLabelerCoMap(pseduo_labeler)).flat_map(
        lambda x: x).filter(lambda x: x not in ['COLLECTING', 'LOW_CONFIDENCE'])

    ds = ds.map(inference.classify).filter(lambda x: x != 'BATCHING')

    ds.print()

    result = env.execute()

else:

    # Create Kafka consumer.
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8')
    )

    us_predictions = []
    ss_predictions = []

    pseudo_data = []
    dump = []

    acc_list = []

    for idx, message in enumerate(consumer):

        # TODO: REMOVE --- ONLY FOR FAST DEBUGGING
        if idx < 1000:
            continue
        if idx > 2000:
            break

        label, text = message.value.split('|||')
        label = int(label)

        text = tokenize(text, STEM)

        us_output = plstream.process_data((idx, label, text))
        ss_output = classifier.classify((idx, label, text))

        if us_output != 'BATCHING':
            us_predictions += us_output
        if ss_output != 'BATCHING':
            ss_predictions += ss_output

        if len(ss_predictions) > 0 or len(us_predictions) > 0:
            temp = [pseduo_labeler.generate_pseudo_label(us_pred, ss_pred)
                    for us_pred, ss_pred in zip_longest(us_predictions, ss_predictions)]

            for t in temp:
                if t not in [['LOW_CONFIDENCE'], ['COLLECTING']]:
                    pseudo_data += t
            acc_list.append(pseduo_labeler.get_model_acc())
            us_predictions, ss_predictions = [], []

        for data in pseudo_data:
            dump.append((data))
            inference.classify((data[0], data[1], data[2]))

        pseudo_data = []

if not PYFLINK:
    print('\n-- UNSUPERVISED MODEL ACCURACY --')
    print(plstream.acc_list)

    print('\n-- SUPERVISED MODEL ACCURACY --')
    print(classifier.acc_list)

    print('\n-- SENTISTREAM ACCURACY --')
    print(acc_list)

    print('\n-- SUPERVISED MODEL ACCURACY ON PSEUDO DATA --')
    print(inference.acc_list)


print('Elapsed Time: ', time() - start)
