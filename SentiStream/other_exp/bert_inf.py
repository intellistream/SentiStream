# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch

from torch.nn.parallel import DataParallel
from kafka import KafkaConsumer
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
            texts.append(text)
            labels.append(int(label))

            if len(labels) >= batch_size:
                input_ids, _ = preprocess(texts)

                logits = model(torch.cat(input_ids, dim=0).to(device))[0]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, axis=1).tolist()

                eval_list += list(zip(ids, preds, labels))

                ids = []
                labels = []
                texts = []
            latency.append(time_ns() - arrival_time)

    consumer.close()

    return (time() - start - 1,  sum(latency)/(len(latency) * 1000000), eval_list)
