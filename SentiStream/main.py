# pylint: disable=import-error
# pylint: disable=no-name-in-module
from time import time, time_ns

import csv

from kafka import KafkaConsumer
from pyflink.datastream import CheckpointingMode, StreamExecutionEnvironment

import config

from inference.classifier import Classifier
from train.supervised import TrainModel
from train.pseudo_labeler import SentimentPseudoLabeler, PseudoLabelerCoMap
from unsupervised_models.plstream import PLStream
from utils import tokenize


def init_train(batch_size=512, lr=0.002, test_size=0.2):
    """
    Initial training of word vector and torch models.

    Args:
        batch_size (int, optional): Batch size for torch model. Defaults to 512.
        lr (float, optional): Learning rate for torch model. Defaults to 0.002.
        test_size (float, optional): Test size for torch model. Defaults to 0.2.
    """
    with open(config.TRAIN_DATA, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            train_data = [[int(row[0]), tokenize(row[1])] for row in reader]
    TrainModel(word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL, init=True,
               vector_size=20, data=train_data, batch_size=batch_size, lr=lr, test_size=test_size)


def stream_process():
    """
    Perform sentiment analysis on stream data.

    Returns:
        tuple: Tuple of thoughput of the system and accuracy and f1 scores of modules and 
        whole system.
    """
    plstream = PLStream(word_vector_algo=config.WORD_VEC_ALGO)
    classifier = Classifier(
        word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL)
    pseduo_labeler = SentimentPseudoLabeler()
    model_trainer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                               ssl_model=config.SSL_MODEL, init=False)

    # if config.PYFLINK:
    #     df = pd.read_csv(config.DATA, names=[
    #         'label', 'review'], nrows=1000)
    #     df['label'] -= 1

    #     # env = StreamExecutionEnvironment.get_execution_environment()
    #     # env.set_parallelism(1)
    #     # env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)

    #     # data_stream = [(idx, label, text)
    #     #                for idx, (label, text) in enumerate(df.values)]

    #     # ds = env.from_collection(collection=data_stream)

    #     # ds = ds.map(lambda x: (x[0], x[1], tokenize(x[2])))

    #     # ds_us = ds.map(plstream.process_data).filter(
    #     #     lambda x: x != config.BATCHING).flat_map(lambda x: x)
    #     # ds_ss = ds.map(classifier.classify).filter(
    #     #     lambda x: x != config.BATCHING).flat_map(lambda x: x)

    #     # ds = ds_us.connect(ds_ss).map(PseudoLabelerCoMap(pseduo_labeler)).flat_map(
    #     #     lambda x: x).filter(lambda x: x not in [config.BATCHING, config.LOW_CONF])

    #     # ds.map(inference.classify).filter(
    #     #     lambda x: x != config.BATCHING)

    #     # # ds.map(model_trainer.update_model)

    #     # ds.print()

    #     # result = env.execute()

    # else:

    # Create Kafka consumer.
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=config.BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8'),
        consumer_timeout_ms=1000  # NOTE: Decrease 1 sec from total time
    )

    us_predictions = []
    ss_predictions = []

    latency = []

    pseudo_data = []

    start = time()

    for idx, message in enumerate(consumer):
        arrival_time = time_ns()

        label, text = message.value.split('|||', 1)
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
            us_predictions, ss_predictions = [], []

            if temp:
                pseudo_data += temp

        if idx % 20000 == 0 and pseudo_data:
            msg = model_trainer.update_model(pseudo_data, 1024)

            if msg == config.FINISHED:
                #     # start = time()
                # plstream.update_word_lists(
                #     pseudo_data)
                #     # print('CLUSTER: ', time() - start)

                #     # start = time()
                plstream.update_word_lists(
                    pseudo_data, temp='t')
                #     # print('SIMPLE: ', time() - start)
                pseudo_data = []

        latency.append(time_ns() - arrival_time)
    consumer.close()

    print(plstream.acc_list)
    # # print(plstream.l_list)
    print(classifier.acc_list)
    print(pseduo_labeler.acc_list)
    return (time() - start - 1, sum(latency)/len(latency)/1000000, plstream.acc_list,
            plstream.f1_list, classifier.acc_list, classifier.f1_list, pseduo_labeler.acc_list,
            pseduo_labeler.f1_list)


if __name__ == '__main__':
    pass
    # batch 128, lr 0.002, te 0.2 - 1%
    # bat 64 0.003 0.3
    # init_train(batch_size=16, lr=0.002, test_size=0.5)  # 1%
    # init_train(batch_size=128, lr=0.001, test_size=0.2)  # 0.5%
    # init_train(batch_size=32, lr=0.005, test_size=0.3)  # 0.1%

    # time_elapsed, us_acc, us_f1, ss_acc, ss_f1, senti_acc, senti_f1 = stream_process()
