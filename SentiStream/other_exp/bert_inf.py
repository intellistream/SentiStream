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

    return (time() - start - 1,  sum(latency)/(len(latency) * 1000000), acc, f1)
