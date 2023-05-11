# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch

from torch.nn.parallel import DataParallel
from kafka import KafkaConsumer
from sklearn.metrics import accuracy_score, f1_score
from time import time, time_ns

import config

from other_exp.utils import preprocess

device = torch.device("cuda")


def get_results(name, batch_size):
    model = torch.load(name + '.pth')
    model = DataParallel(model)
    model.eval()

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
    latency = []

    texts = []
    labels = []

    start = time()
    with torch.no_grad():
        for message in consumer:
            arrival_time = time_ns()

            label, text = message.value.split('|||', 1)
            texts.append(text)
            labels.append(int(label))

            if len(labels) >= batch_size:
                input_ids, _ = preprocess(texts)

                logits = model(torch.cat(input_ids, dim=0).to(device))[0]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, axis=1).tolist()

                latency.append(time_ns() - arrival_time)
                acc.append(accuracy_score(preds, labels))
                f1.append(f1_score(preds, labels))

                labels = []
                texts = []

                # print(acc[-1], f1[-1])

    consumer.close()

    return (time() - start,  sum(latency)/(len(latency) * 1000000), acc, f1)


# BERT 1%

# [0.8525, 0.8425, 0.8425, 0.8375, 0.8375, 0.836, 0.8375, 0.851, 0.837, 0.832, 0.8445, 0.831, 0.848, 0.831, 0.836, 0.8455, 0.84, 0.851, 0.833, 0.8425, 0.8375, 0.834, 0.8375, 0.8505, 0.8445, 0.85, 0.848, 0.8415, 0.857, 0.842, 0.8615, 0.838, 0.841, 0.841, 0.844, 0.8355, 0.8415, 0.8345, 0.852, 0.815, 0.8205, 0.819, 0.8195, 0.8225, 0.815, 0.817, 0.8105, 0.8165, 0.8095, 0.8045,
#     0.8085, 0.798, 0.8345, 0.823, 0.8105, 0.815, 0.8045, 0.809, 0.811, 0.7965, 0.8235, 0.8275, 0.8095, 0.76, 0.537, 0.5295, 0.521, 0.518, 0.537, 0.5065, 0.5205, 0.514, 0.52, 0.5255, 0.5155, 0.529, 0.523, 0.5125, 0.515, 0.5215, 0.5205, 0.504, 0.524, 0.5215, 0.519, 0.5175, 0.5275, 0.5075, 0.5325, 0.511, 0.519, 0.5225, 0.5265, 0.5425, 0.5175, 0.5355, 0.5125, 0.5201640464798359]
