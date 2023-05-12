# pylint: disable=import-error
# pylint: disable=no-name-in-module
from time import time, time_ns

import csv

from kafka import KafkaConsumer

import config

from inference.classifier import Classifier
from train.supervised import TrainModel
from train.pseudo_labeler import SentimentPseudoLabeler
from unsupervised_models.plstream import PLStream
from utils import tokenize


def init_train(batch_size=512, lr=0.002, test_size=0.2, min_count=5):
    """
    Initial training of word vector and torch models.

    Args:
        batch_size (int, optional): Batch size for torch model. Defaults to 512.
        lr (float, optional): Learning rate for torch model. Defaults to 0.002.
        test_size (float, optional): Test size for torch model. Defaults to 0.2.
    """
    with open(config.TRAIN_DATA, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        train_data = [[int(row[0]), tokenize(row[1])] for row in reader]
    TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
               ssl_model=config.SSL_MODEL, init=True, vector_size=20,
               data=train_data, batch_size=batch_size, lr=lr,
               test_size=test_size, min_count=min_count)


def stream_process(lower_thresh, update_thresh, update):
    """
    Perform sentiment analysis on stream data.

    Returns:
        tuple: Tuple of thoughput of the system and accuracy and f1 scores of modules and 
        whole system.
    """
    SentimentPseudoLabeler.FIXED_NEG_THRESHOLD = lower_thresh
    SentimentPseudoLabeler.FIXED_POS_THRESHOLD = lower_thresh

    plstream = PLStream(
        word_vector_algo=config.WORD_VEC_ALGO)
    classifier = Classifier(
        word_vector_algo=config.WORD_VEC_ALGO, ssl_model=config.SSL_MODEL)
    pseduo_labeler = SentimentPseudoLabeler()
    model_trainer = TrainModel(word_vector_algo=config.WORD_VEC_ALGO,
                               ssl_model=config.SSL_MODEL, init=False)

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

    senti_latency = []
    us_latency = []
    ss_latency = []
    pseudo_data = []

    start = time()

    for idx, message in enumerate(consumer):
        # arrival_time = message.timestamp ## THIS IS MESSAGE PRODUCED TIMESTAMP
        arrival_time = time_ns()

        label, text = message.value.split('|||', 1)
        label = int(label)

        text = tokenize(text)

        us_output = plstream.process_data((idx, label, text))
        us_latency.append(time_ns() - arrival_time)

        ss_output = classifier.classify((idx, label, text))
        ss_latency.append(time_ns() - arrival_time - us_latency[-1])

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

        if idx % update_thresh == 0 and pseudo_data:
            msg = model_trainer.update_model(pseudo_data, 712)
            plstream.update_word_lists(pseudo_data, update=update)

            if msg == config.FINISHED:
                pseudo_data = []

        senti_latency.append(time_ns() - arrival_time)
    consumer.close()

    return (time() - start - 1, sum(senti_latency)/(len(senti_latency) * 1000000),
            sum(us_latency) / (len(us_latency) * 1000000),
            sum(ss_latency) / (len(ss_latency) * 1000000), plstream.acc_list,
            plstream.f1_list, classifier.acc_list, classifier.f1_list,
            pseduo_labeler.acc_list, pseduo_labeler.f1_list)
