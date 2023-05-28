# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch

from kafka import KafkaConsumer
from time import time, time_ns

import config

from other_exp.utils import tokenize
from utils import get_average_word_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_preds(clf, w2v, batch_size):
    wv_model = w2v
    model = clf
    model.eval()

    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=config.BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8'),
        consumer_timeout_ms=1000  # NOTE: Decrease 1 sec from total time
    )

    eval_list = []
    latency = []

    ids = []
    texts = []
    labels = []

    start = time()

    with torch.no_grad():
        for message in consumer:
            arrival_time = time_ns()

            id, label, text = message.value.split('|||')

            ids.append(id)
            texts.append(tokenize(text))
            labels.append(int(label))

            if len(labels) >= batch_size or id == '-1':

                texts = get_average_word_embeddings(wv_model, texts)

                preds = model(torch.from_numpy(texts).to(device))
                preds = preds.ge(0.5).long().view(-1).tolist()

                eval_list += list(zip(ids, preds, labels))

                ids = []
                labels = []
                texts = []
            latency.append(time_ns() - arrival_time)

    consumer.close()

    return (time() - start - 1,  sum(latency)/(len(latency) * 1000000), eval_list)
