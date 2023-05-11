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


def test_sentistream(percent, batch_size, lr, test_size, min_count=5, use_pretrained=True,
                     name='data', lower_thresh=0.8, update_thresh=20000, update_lex=True):
    """
    Evaluate performance metrics of SentiStream.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
        batch_size (_type_): Batch size for initial training of torch model.
        lr (_type_): Learning rate for initial training of torch model.
        test_size (_type_): Test size for initial training of torch model.
        use_pretrained (bool, optional): Flag to use pretrained best model. Defaults to True.
        name (str, optional): Name of dataset to be tested.
        lower_thresh (float, optional): Lower threshold for stream merge.
        update_thresh (int, optional): Update threshold for updating models.
    """
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    if use_pretrained:
        load_pretrained_models(percent, name)
    else:
        init_train(batch_size=batch_size, lr=lr,
                   test_size=test_size, min_count=min_count)

    time, latency, us_acc, us_f1, ss_acc, ss_f1, senti_acc, senti_f1 = stream_process(
        lower_thresh, update_thresh, update_lex)
    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    print('Avg US_ACC: ', sum(us_acc)/len(us_acc))
    print('Avg US_F1: ', sum(us_f1)/len(us_f1))
    print('Avg SS_ACC: ', sum(ss_acc)/len(ss_acc))
    print('Avg SS_F1: ', sum(ss_f1)/len(ss_f1))
    print('Avg Senti_ACC: ', sum(senti_acc)/len(senti_acc))
    print('Avg Senti_F1: ', sum(senti_f1)/len(senti_f1))

    with open(f'outputs/output_{name}_{percent}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['us_acc', 'us_f1', 'ss_acc',
                        'ss_f1', 'senti_acc', 'senti_f1'])

        for row in zip(*[us_acc, us_f1, ss_acc, ss_f1, senti_acc, senti_f1]):
            writer.writerow(row)


# 0.5 %
# combined
print('\n--Combined Dataset--\n')
test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
                 use_pretrained=True, name='data', lower_thresh=0.7, update_thresh=20000)

# yelp
print('\n--Yelp Dataset--\n')
test_sentistream(percent='0_5', batch_size=64, lr=0.01, test_size=0.2, min_count=3,
                 use_pretrained=True, name='yelp', lower_thresh=0.8, update_thresh=20000)

# imdb
print('\n--IMDb Dataset--\n')
test_sentistream(percent='0_5', batch_size=32, lr=0.003, test_size=0.2, min_count=3,
                 use_pretrained=True, name='imdb', lower_thresh=0.5, update_thresh=10000)

# sst-2
print('\n--SST-2 Dataset--\n')
test_sentistream(percent='0_5', batch_size=32, lr=0.02, test_size=0.2,
                 min_count=2, use_pretrained=True, name='sst', lower_thresh=0.8,
                 update_thresh=10000, update_lex=False)

# ---------------------------------------------------------------------------------------------- #

# OTHERS

# # COMBINED DATA
# # Model trained on 1% data
# test_sentistream(percent='1', batch_size=256, lr=0.005, test_size=0.2, min_count=5,
#                  use_pretrained=True, name='data', lower_thresh=0.7, update_thresh=20000)

# # Model trained on 0.1% data
# test_sentistream(percent='0_1', batch_size=16, lr=0.005, test_size=0.3, min_count=5,
#                  use_pretrained=True, name='data', lower_thresh=0.7, update_thresh=20000)


# YELP
# Model trained on 1% data
# test_sentistream(percent='1', batch_size=128, lr=0.004, test_size=0.2, min_count=5,
#                  use_pretrained=True, name='yelp', lower_thresh=0.8, update_thresh=20000)

# # Model trained on 0.1% data
# test_sentistream(percent='0_1', batch_size=16, lr=0.02, test_size=0.3, min_count=5,
#                  use_pretrained=True, name='yelp', lower_thresh=0.6, update_thresh=20000)


# # IMDB
# # Model trained on 1% data
# test_sentistream(percent='1', batch_size=64, lr=0.003, test_size=0.2, min_count=3,
#                  use_pretrained=True, name='imdb', lower_thresh=0.6, update_thresh=10000)

# # Model trained on 0.5% data


# # Model trained on 0.1% data
# test_sentistream(percent='0_1', batch_size=16, lr=0.005, test_size=0.3, min_count=2,
#                  use_pretrained=True, name='imdb', lower_thresh=0.7, update_thresh=10000)


# SST
# Model trained on 1% data
# test_sentistream(percent='1', batch_size=64, lr=0.01, test_size=0.2, min_count=2,
#                  use_pretrained=True, name='sst', lower_thresh=0.9, update_thresh=10000)

# # Model trained on 0.1% data
# test_sentistream(percent='0_1', batch_size=16, lr=0.003, test_size=0.5, min_count=3,
#                  use_pretrained=True, name='sst', lower_thresh=0.5, update_thresh=10000)
