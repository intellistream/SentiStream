# pylint: disable=import-error
# pylint: disable=no-name-in-module
import random

from kafka import KafkaConsumer
from sklearn.metrics import accuracy_score, f1_score

import config

random.seed(42)


def test(batch_size):
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=config.BOOTSTRAP_SERVER,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode('utf-8'),
        consumer_timeout_ms=1000  # NOTE: Decrease 1 sec from total time
    )

    acc = []
    f1 = []
    labels = []

    for message in consumer:
        labels.append(int(message.value.split('|||', 1)[0]))

        if len(labels) >= batch_size:
            pred = [random.randint(0, 1) for i in range(10000)]
            acc.append(accuracy_score(pred, labels))
            f1.append(f1_score(pred, labels))
            labels = []

    consumer.close()
    return acc, f1
