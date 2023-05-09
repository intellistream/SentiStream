# pylint: disable=import-error
# pylint: disable=no-name-in-module
import csv
import os
import torch

from gensim.models import Word2Vec

import config

from main import stream_process, init_train
from kafka_producer import create_stream


if not os.path.exists('outputs'):
    os.makedirs('outputs')


def load_pretrained_models(percent, dataset):
    """
    Rename pretrained word vector and torch models with best weights for current inference. To
    avoid overwriting while updatingm models.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
    """
    Word2Vec.load(
        f'trained_models/best_{dataset}_{percent}.model').save(config.SSL_WV)
    torch.save(torch.load(
        f'trained_models/best_{dataset}_{percent}.pth'), config.SSL_CLF)


def test_sentistream(percent, batch_size, lr, test_size, use_pretrained=True, name='data'):
    """
    Evaluate performance metrics of SentiStream.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
        batch_size (_type_): Batch size for initial training of torch model.
        lr (_type_): Learning rate for initial training of torch model.
        test_size (_type_): Test size for initial training of torch model.
        use_pretrained (bool, optional): Flag to use pretrained best model. Defaults to True.
    """
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    if use_pretrained:
        load_pretrained_models(percent, name)
    else:
        init_train(batch_size=batch_size, lr=lr, test_size=test_size)

    time, latency, us_acc, us_f1, ss_acc, ss_f1, senti_acc, senti_f1 = stream_process()
    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    with open(f'outputs/output_{name}_{percent}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['us_acc', 'us_f1', 'ss_acc',
                        'ss_f1', 'senti_acc', 'senti_f1'])

        for row in zip(*[us_acc, us_f1, ss_acc, ss_f1, senti_acc, senti_f1]):
            writer.writerow(row)


# # COMBINED DATA
# Model trained on 1% data
# test_sentistream('1', 512, 0.002, 0.2, True)

# # Model trained on 0.5% data
# test_sentistream('0_5', 128, 0.01, 0.2, True)

# # Model trained on 0.1% data
# test_sentistream('0_1', 32, 0.005, 0.3, True)


# # YELP
# # Model trained on 1% data
test_sentistream('1', 128, 0.002, 0.2, True, 'yelp')

# # # Model trained on 0.5% data
# test_sentistream('0_5', 64, 0.003, 0.3, True, 'yelp')

# # Model trained on 0.1% data
# test_sentistream('0_1', 16, 0.002, 0.3, True, 'yelp')


# # IMDB
# # Model trained on 1% data
# test_sentistream('1', 64, 0.002, 0.2, True, 'imdb')

# # Model trained on 0.5% data
# test_sentistream('0_5', 32, 0.001, 0.2, True, 'imdb')

# # Model trained on 0.1% data
# test_sentistream('0_1', 16, 0.001, 0.4, True, 'imdb')


# SST
# Model trained on 1% data
# test_sentistream('1', 64, 0.002, 0.3, False, 'sst')

# # Model trained on 0.5% data
# test_sentistream('0_5', 64, 0.001, 0.3, False, 'sst')

# # Model trained on 0.1% data
# test_sentistream('0_1', 16, 0.002, 0.5, False, 'sst')
