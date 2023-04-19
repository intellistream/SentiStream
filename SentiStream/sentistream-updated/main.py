# pylint: disable=import-error
# pylint: disable=no-name-in-module

# TORCH JIT ?
# TODO: FIX CLASS IMBALANCE IN INITIAL TRAINING - DONE - REWATCH

from itertools import zip_longest
from time import time

import pandas as pd

from kafka import KafkaConsumer
from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment

import config
from train.supervised import TrainModel
from train.pseudo_labeler import SentimentPseudoLabeler, PseudoLabelerCoMap
from unsupervised_models.plstream import PLStream
from inference.classifier import Classifier
from utils import tokenize


start = time()

# ---------------- Initial training of classifier ----------------
# Train word vector {Word2Vec or FastText} on {nrows=1000} data and get word embeddings, then use
# that embedding to train NN sentiment classifier {ANN or HAN}
TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
           ssl_model=config.SSL_MODEL, init=True, nrows=5600, vector_size=20)


# ---------------- Generate pesudo labels ----------------

df = pd.read_csv(config.DATA, names=[
    'label', 'review'], nrows=1000)
df['label'] -= 1

plstream = PLStream(word_vector_algo=config.WORD_VEC_ALGO)
classifier = Classifier(
    word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL)
pseduo_labeler = SentimentPseudoLabeler()
inference = Classifier(word_vector_algo=config.WORD_VEC_ALGO,
                       ssl_model=config.SSL_MODEL, is_eval=True)
model_trainer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                           ssl_model=config.SSL_MODEL, init=False,
                           acc_threshold=0.9)

if config.PYFLINK:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    data_stream = [(idx, label, text)
                   for idx, (label, text) in enumerate(df.values)]

    ds = env.from_collection(collection=data_stream)

    ds = ds.map(lambda x: (x[0], x[1], tokenize(x[2])))

    ds_us = ds.map(plstream.process_data).filter(
        lambda x: x != config.BATCHING).flat_map(lambda x: x)
    ds_ss = ds.map(classifier.classify).filter(
        lambda x: x != config.BATCHING).flat_map(lambda x: x)

    ds = ds_us.connect(ds_ss).map(PseudoLabelerCoMap(pseduo_labeler)).flat_map(
        lambda x: x).filter(lambda x: x not in [config.BATCHING, config.LOW_CONF])

    ds.map(inference.classify).filter(
        lambda x: x != config.BATCHING)  # TODO: ACC

    # ds.map(model_trainer.update_model)

    ds.print()

    result = env.execute()

else:

    # Create Kafka consumer.
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=config.BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8')
    )

    us_predictions = []
    ss_predictions = []

    pseudo_data = []
    dump = []

    acc_list = []

    start = time()

    # TODO: DO IT PARALLELy
    for idx, message in enumerate(consumer):

        # TODO: REMOVE --- ONLY FOR FAST DEBUGGING
        if idx < 5600:
            continue
        if idx > 20000:
            break

        label, text = message.value.split('|||')
        label = int(label)

        text = tokenize(text)

        us_output = plstream.process_data((idx, label, text))

        ss_output = classifier.classify((idx, label, text))

        if us_output != config.BATCHING:
            us_predictions += us_output
        if ss_output != config.BATCHING:
            ss_predictions += ss_output

        if len(ss_predictions) > 0 or len(us_predictions) > 0:
            temp = [pseduo_labeler.generate_pseudo_label(us_pred, ss_pred)
                    for us_pred, ss_pred in zip_longest(us_predictions, ss_predictions)]

            for t in temp:
                if t not in [[config.LOW_CONF], [config.BATCHING]]:
                    pseudo_data += t
            acc_list.append(pseduo_labeler.get_model_acc())
            us_predictions, ss_predictions = [], []

        for data in pseudo_data:
            dump.append(data[1:])
            inference.classify((data[0], data[1], data[2]))

        pseudo_data = []

        if idx % 1000 == 0:
            if dump:
                message = model_trainer.update_model(dump, 0.4, 0.2)

                if message == config.FINISHED:
                    dump = []

print(
    f'\n\nBOTH SAME - CORRECT: {pseduo_labeler.us_ss_same_crct}, WRONG: {pseduo_labeler.us_ss_same_wrng}')
print(
    f'SS CORRECT: {pseduo_labeler.ss_crct}, US_CORRECT: {pseduo_labeler.us_crct}')
# print(pseduo_labeler.both_same_crct_conf)

# print(pseduo_labeler.both_same_wrong_conf)

# NOTE: MOST OF BOTH SAME RESULTS IN CORRECT PREDICTION
# NOTE: EVEN HIGH CONFIDENT LABELS HAVE 10% WRONG LABELS (SAME WEIGHT FOR BOTH and set conf 0.5)
# NOTE: HAVING 0.3 as conf thresh remove most of the noise in pseudo labels.


if not config.PYFLINK:
    print('\n-- UNSUPERVISED MODEL ACCURACY --')
    print(plstream.acc_list)
    print(sum(plstream.acc_list)/len(plstream.acc_list))

    print('\n-- SUPERVISED MODEL ACCURACY --')
    print(classifier.acc_list)
    print(sum(classifier.acc_list)/len(classifier.acc_list))

    print('\n-- SENTISTREAM ACCURACY --')
    print(acc_list)
    print(sum([x for x in acc_list if x])/len([x for x in acc_list if x]))

    print('\n-- SUPERVISED MODEL ACCURACY ON PSEUDO DATA --')
    print(inference.acc_list)
    print(sum(inference.acc_list)/len(inference.acc_list))

print('Elapsed Time: ', time() - start)
