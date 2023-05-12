# pylint: disable=import-error
# pylint: disable=no-name-in-module

import csv

import config

from kafka_producer import create_stream
from other_exp.bert_training import train
from other_exp.bert_inf import get_results
from other_exp.random_pred import test
from other_exp.self_learning import test_sl


def test_bert(percent, batch_size, epochs, lr, name, inf_batch_size):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    # comment this if model is already trained
    train(batch_size, epochs, lr, 'bert_' + name + '_' + percent)

    time, latency, acc, f1 = get_results(
        'bert_' + name + '_' + percent, inf_batch_size)

    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    print('Avg ACC: ', sum(acc)/len(acc))
    print('Avg F1: ', sum(f1)/len(f1))

    with open(f'outputs/bt_output_{name}_{percent}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['acc', 'f1'])

        for row in zip(*[acc, f1]):
            writer.writerow(row)


def test_self_learning(percent, name):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    time, latency, acc, f1 = test_sl()

    print('Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')
    print('Avg ACC: ', sum(acc)/len(acc))
    print('Avg F1: ', sum(f1)/len(f1))

    with open(f'outputs/sl_output_{name}_{percent}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['acc', 'f1'])

        for row in zip(*[acc, f1]):
            writer.writerow(row)


def test_random(percent, batch_size, name):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    create_stream()

    acc, f1 = test(batch_size)

    print('Avg ACC: ', sum(acc)/len(acc))
    print('Avg F1: ', sum(f1)/len(f1))

    with open(f'outputs/rn_output_{name}_{percent}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['acc', 'f1'])

        for row in zip(*[acc, f1]):
            writer.writerow(row)


# BERT
# # 0.5 %
# # combined
# print('\n--Combined Dataset--\n')
# test_bert(percent='0_5', batch_size=64, epochs=10,
        #   lr=5e-5, name='data', inf_batch_size=8)

# # yelp
# print('\n--Yelp Dataset--\n')
# test_bert(percent='0_5', batch_size=32, epochs=3,
#           lr=5e-5, name='yelp', inf_batch_size=8)

# # imdb
# print('\n--IMDb Dataset--\n')
# test_bert(percent='0_5', batch_size=64, epochs=5,
#           lr=5e-5, name='imdb', inf_batch_size=8)

# # sst-2
# print('\n--SST-2 Dataset--\n')
# test_bert(percent='0_5', batch_size=64, epochs=3,
#           lr=5e-5, name='sst', inf_batch_size=8)


# # 1 %
# combined
# print('\n--Combined Dataset--\n')
# test_bert(percent='1', batch_size=64, epochs=10,
#           lr=5e-5, name='data', inf_batch_size=8)

# # yelp
# print('\n--Yelp Dataset--\n')
# test_bert(percent='1', batch_size=64, epochs=10,
#           lr=5e-5, name='yelp', inf_batch_size=8)

# # imdb
# print('\n--IMDb Dataset--\n')
# test_bert(percent='1', batch_size=64, epochs=10,
#           lr=5e-5, name='imdb', inf_batch_size=8)

# # sst-2
# print('\n--SST-2 Dataset--\n')
# test_bert(percent='1', batch_size=32, epochs=3,
#           lr=5e-6, name='sst', inf_batch_size=8)


# ------------------------------------------------------------------------ #

# SELF LEARNING
# 0.5 %
# combined
# print('\n--Combined Dataset--\n')
# test_self_learning('0_5', 'data')

# yelp
# print('\n--Yelp Dataset--\n')
# test_self_learning('0_5', 'yelp')

# imdb
# print('\n--IMDb Dataset--\n')
# test_self_learning('0_5', 'imdb')

# sst-2
# print('\n--SST-2 Dataset--\n')
# test_self_learning('0_5', 'sst')


# 1 %
# combined
# print('\n--Combined Dataset--\n')
# test_self_learning('1', 'data')

# yelp
# print('\n--Yelp Dataset--\n')
# test_self_learning('1', 'yelp')

# imdb
# print('\n--IMDb Dataset--\n')
# test_self_learning('1', 'imdb')

# sst-2
# print('\n--SST-2 Dataset--\n')
# test_self_learning('1', 'sst')

# ------------------------------------------------------------------------ #


# RANDOM
# # 0.5 %
# # combined
# print('\n--Combined Dataset--\n')
# test_random(percent='0_5', batch_size=10000, name='data')

# # yelp
# print('\n--Yelp Dataset--\n')
# test_random(percent='0_5', batch_size=10000, name='yelp')

# # imdb
# print('\n--IMDb Dataset--\n')
# test_random(percent='0_5', batch_size=10000, name='imdb')

# # sst-2
# print('\n--SST-2 Dataset--\n')
# test_random(percent='0_5', batch_size=10000, name='sst')


# # 1 %
# # combined
# print('\n--Combined Dataset--\n')
# test_random(percent='1', batch_size=10000, name='data')

# # yelp
# print('\n--Yelp Dataset--\n')
# test_random(percent='1', batch_size=10000, name='yelp')

# # imdb
# print('\n--IMDb Dataset--\n')
# test_random(percent='1', batch_size=10000, name='imdb')

# # sst-2
# print('\n--SST-2 Dataset--\n')
# test_random(percent='1', batch_size=10000, name='sst')
