# pylint: disable=import-error
# pylint: disable=no-name-in-module
from time import time

import csv
import pandas as pd

from kafka import KafkaConsumer
from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment

import config

from inference.classifier import Classifier
from train.supervised import TrainModel
from train.pseudo_labeler import SentimentPseudoLabeler, PseudoLabelerCoMap
from unsupervised_models.plstream import PLStream
from utils import tokenize

# ---------------- Initial training of classifier ----------------
with open(config.TRAIN_DATA, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        train_data = [[int(row[0]), tokenize(row[1])] for row in reader]
start = time()
TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
           ssl_model=config.SSL_MODEL, init=True, vector_size=20, data=train_data)

# # ---------------- Stream Processing ----------------
plstream = PLStream(word_vector_algo=config.WORD_VEC_ALGO)
classifier = Classifier(
    word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL)
pseduo_labeler = SentimentPseudoLabeler()
model_trainer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                           ssl_model=config.SSL_MODEL, init=False)

if config.PYFLINK:
    df = pd.read_csv(config.DATA, names=[
        'label', 'review'], nrows=1000)
    df['label'] -= 1

    # env = StreamExecutionEnvironment.get_execution_environment()
    # env.set_parallelism(1)
    # env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    # data_stream = [(idx, label, text)
    #                for idx, (label, text) in enumerate(df.values)]

    # ds = env.from_collection(collection=data_stream)

    # ds = ds.map(lambda x: (x[0], x[1], tokenize(x[2])))

    # ds_us = ds.map(plstream.process_data).filter(
    #     lambda x: x != config.BATCHING).flat_map(lambda x: x)
    # ds_ss = ds.map(classifier.classify).filter(
    #     lambda x: x != config.BATCHING).flat_map(lambda x: x)

    # ds = ds_us.connect(ds_ss).map(PseudoLabelerCoMap(pseduo_labeler)).flat_map(
    #     lambda x: x).filter(lambda x: x not in [config.BATCHING, config.LOW_CONF])

    # ds.map(inference.classify).filter(
    #     lambda x: x != config.BATCHING)

    # # ds.map(model_trainer.update_model)

    # ds.print()

    # result = env.execute()

else:

    # # Create Kafka consumer.
    # consumer = KafkaConsumer(
    #     config.KAFKA_TOPIC,
    #     bootstrap_servers=config.BOOTSTRAP_SERVER,
    #     auto_offset_reset='earliest',
    #     enable_auto_commit=True,
    #     value_deserializer=lambda x: x.decode('utf-8')
    # )

    consumer = []

    with open(config.DATA, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            consumer.append(f'{row[0]}|||{str(row[1])}')

    us_predictions = []
    ss_predictions = []

    pseudo_data = []
    acc_list = []

    start = time()
    for idx, message in enumerate(consumer):
        # TODO: DELETE WHEN USING KAFKA---
        # if idx < 80000:
        #     continue
        # if idx > 100000:
        #     break

        # idx -= 80000

        # label, text = message.value.split('|||', 1)
        label, text = message.split('|||', 1)
        label = int(label)

        text = tokenize(text)

        us_output = plstream.process_data((idx, label, text))
        ss_output = classifier.classify((idx, label, text))

        if us_output:
            us_predictions += us_output
        if ss_output:
            ss_predictions += ss_output

        if ss_predictions or us_predictions:
            temp = pseduo_labeler.generate_pseudo_label(
                us_predictions, ss_predictions)
            acc_list.append(pseduo_labeler.get_model_acc())
            us_predictions, ss_predictions = [], []

            if temp:
                pseudo_data += temp

        if idx % 20000 == 0 and pseudo_data:
            msg = model_trainer.update_model(pseudo_data, 1024)

            if msg == config.FINISHED:
                #     # start = time()
                #     # plstream.update_word_lists(
                #     #     pseudo_data)
                #     # print('CLUSTER: ', time() - start)

                #     # start = time()
                plstream.update_word_lists(
                    pseudo_data, temp='t')
                #     # print('SIMPLE: ', time() - start)
                pseudo_data = []

stop = time()
if not config.PYFLINK:
    print('\n-- UNSUPERVISED MODEL ACCURACY --')
    print('--with simple update--')
    print(plstream.acc_list)
    print('AVG ACC SO FAR: ', sum(plstream.acc_list) /
          len(plstream.acc_list))

    # print('--without--')
    # print(plstream.ts_list)
    # print('AVG ACC SO FAR: ', sum(plstream.ts_list) /
    #       len(plstream.ts_list))

    # print('--with clustering --')
    # print(plstream.l_list)
    # print('AVG ACC SO FAR: ', sum(plstream.l_list) /
    #       len(plstream.l_list))

    # print('--0.4--')
    # print(plstream.lt_list)
    # print('AVG ACC SO FAR: ', sum(plstream.lt_list) /
    #       len(plstream.lt_list))

    print('\n-- SEMI-SUPERVISED MODEL ACCURACY --')
    print(classifier.acc_list)
    print('AVG ACC SO FAR: ', sum(classifier.acc_list)/len(classifier.acc_list))

    acc_list = [x for x in acc_list if x]
    print('\n-- SENTISTREAM ACCURACY --')
    print(acc_list)
    print('AVG ACC SO FAR: ', sum(
        acc_list)/len(acc_list))

print('Elapsed Time: ', stop - start)
