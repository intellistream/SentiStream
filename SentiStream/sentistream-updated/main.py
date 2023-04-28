# pylint: disable=import-error
# pylint: disable=no-name-in-module

# NOTE: NOT USING TEMPORAL TREND DETECTION

from time import time

import pandas as pd
import matplotlib.pyplot as plt

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
mem_buffer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                        ssl_model=config.SSL_MODEL, init=True, vector_size=20).mem_buffer

# ---------------- Generate pesudo labels ----------------
plstream = PLStream(word_vector_algo=config.WORD_VEC_ALGO)
classifier = Classifier(
    word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL)
pseduo_labeler = SentimentPseudoLabeler()
inference = Classifier(word_vector_algo=config.WORD_VEC_ALGO,
                       ssl_model=config.SSL_MODEL, is_eval=True)
model_trainer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                           ssl_model=config.SSL_MODEL, init=False,
                           acc_threshold=0.9, mem_buffer=mem_buffer)

if config.PYFLINK:
    df = pd.read_csv(config.DATA, names=[
        'label', 'review'], nrows=1000)
    df['label'] -= 1

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

    # # Create Kafka consumer.
    # consumer = KafkaConsumer(
    #     config.KAFKA_TOPIC,
    #     bootstrap_servers=config.BOOTSTRAP_SERVER,
    #     auto_offset_reset='earliest',
    #     enable_auto_commit=True,
    #     value_deserializer=lambda x: x.decode('utf-8')
    # )

    consumer = []
    import csv
    with open(config.DATA, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            consumer.append(f'{int(row[0]) - 1}|||{str(row[1])}')

    # TODO: USE MULTIPROCESSING QUEUE TO THIS
    plstream.create_lexicon()

    us_predictions = []
    ss_predictions = []

    pseudo_data = []
    dump = []

    acc_list = []

    # TODO: DO IT PARALLELy
    start = time()
    for idx, message in enumerate(consumer):  # 515099
        # Set MAX LIMIT (515099) here... else consumer will continously wait for new data to
        # arrive and not finish the program.
        # TODO: DELETE WHEN USING KAFKA---
        if idx < 5600:
            continue
        # if idx > 500000:
        #     break

        # label, text = message.value.split('|||', 1)
        label, text = message.split('|||', 1)
        label = int(label)

        text = tokenize(text)

        us_output = plstream.process_data((idx, label, text))

        ss_output = classifier.classify((idx, label, text))

        if us_output != config.BATCHING:
            us_predictions += us_output
        if ss_output != config.BATCHING:
            ss_predictions += ss_output

        if len(ss_predictions) > 0 or len(us_predictions) > 0:
            temp = pseduo_labeler.generate_pseudo_label(
                us_predictions, ss_predictions)
            acc_list.append(pseduo_labeler.get_model_acc())
            us_predictions, ss_predictions = [], []

            if temp and temp != [config.BATCHING]:
                for data in temp:
                    dump.append(data[1:])
        #             inference.classify(
        #                 (data[0], data[1], data[2]), batch_size=len(temp))
            temp = []
#         # dump.append([label, text]) # DEBUG - WITH GROUND TRUTH

#         # TODO: CHANGE COND FOR MULTIPROCESSING AS IT WONT WORK
#         # if len(dump) > 10000:
        if idx % 20000 == 0:
            if dump:
                message = model_trainer.update_model(dump, 0.4, 0.2)
                plstream.create_lexicon()
                message_us = plstream.update_word_lists(dump)
                if message == config.FINISHED:
                    dump = []
    #                 # pass

# print(
#     f'\n\nFLEXMATCH - BOTH PREDICTIONS ARE SAME - CORRECT: {pseduo_labeler.us_ss_same_crct}, WRONG: {pseduo_labeler.us_ss_same_wrng}')
# print(
#     f'\n\nWITHOUT FLEXMATCH - BOTH PREDICTIONS ARE SAME - CORRECT: {pseduo_labeler.us_ss_same_crct_aft}, WRONG: {pseduo_labeler.us_ss_same_wrng_aft}')


# print(
#     f'\n TOTAL LABELS BY GROUND TRUTH, US PRED, SS PRED --- POS  {pseduo_labeler.ttl_true_pos}, {pseduo_labeler.ttl_us_pos}, {pseduo_labeler.ttl_ss_pos},,, NEG {pseduo_labeler.ttl_true_neg}, {pseduo_labeler.ttl_us_neg}, {pseduo_labeler.ttl_ss_neg}')

# print(
#     f'FLEXMATCH - PSEUDO (CORRECT, WRONG) --- POS {pseduo_labeler.pseudo_pos_crct}, {pseduo_labeler.pseudo_pos_wrng},, NEG {pseduo_labeler.pseudo_neg_crct}, {pseduo_labeler.pseudo_neg_wrng}')

# print(
#     f'WITHOUT FLEXMATCH - PSEUDO (CORRECT, WRONG) --- POS {pseduo_labeler.pseudo_pos_crct_aft}, {pseduo_labeler.pseudo_pos_wrng_aft},, NEG {pseduo_labeler.pseudo_neg_crct_aft}, {pseduo_labeler.pseudo_neg_wrng_aft}')


# # NOTE: MOST OF BOTH SAME RESULTS IN CORRECT PREDICTION
# # NOTE: EVEN HIGH CONFIDENT LABELS HAVE < 10% WRONG LABELS (SAME WEIGHT FOR BOTH and set conf 0.5)

# # NOTE: HAN IS BETTER THAN ANN ---- NOT MUCH DIFF IN PREDICTIONS AT EARLY STAGE,, BUT LOSS IS LOW,
# #  AND GENERATES HIGH CONF -- CORRECT PSEUDO LABELS THAN ANN

if not config.PYFLINK:
    # print('\n-- UNSUPERVISED MODEL ACCURACY --')
    # # print('--text sim--')
    # print(plstream.acc_list)
    # print('AVG ACC SO FAR: ', sum(plstream.acc_list) /
    #       len(plstream.acc_list))

    # print('--default--')
    # print(plstream.ts_list)
    # print('AVG ACC SO FAR: ', sum(plstream.ts_list) /
    #       len(plstream.ts_list))

    # print('--lexicon--')
    # print(plstream.l_list)
    # print('AVG ACC SO FAR: ', sum(plstream.l_list) /
    #       len(plstream.l_list))

    # print('--lexicon + ts--')
    # print(plstream.lt_list)
    # print('AVG ACC SO FAR: ', sum(plstream.lt_list) /
    #       len(plstream.lt_list))
    # plt.plot([x*5000 for x in range(len(plstream.acc_list))],
    #          plstream.acc_list, label='default', marker='o')

    # plt.legend()
    # plt.savefig('sentistream.png')

    # print(f'\nORIG - {plstream.count} TS - {plstream.count2}')

    # print('\n-- SUPERVISED MODEL ACCURACY --')
    # print(classifier.acc_list)
    # print('AVG ACC SO FAR: ', sum(classifier.acc_list)/len(classifier.acc_list))

    print('\n-- SENTISTREAM ACCURACY --')
    print(acc_list)
    print('AVG ACC SO FAR: ', sum(
        [x for x in acc_list if x])/len([x for x in acc_list if x]))

#     print('\n-- SUPERVISED MODEL ACCURACY ON PSEUDO DATA --')
#     print(inference.acc_list)
#     print('AVG ACC SO FAR: ', sum(inference.acc_list)/len(inference.acc_list))


print('Elapsed Time: ', time() - start)
