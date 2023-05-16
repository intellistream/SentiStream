# pylint: disable=import-error
# pylint: disable=no-name-in-module

import csv

from sklearn.metrics import accuracy_score, f1_score

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
    # train(batch_size, epochs, lr, 'bert_' + name + '_' + percent)

    time, latency, eval_list = get_results(
        'bert_' + name + '_' + percent, inf_batch_size)

    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    yelp, imdb, sst = [], [], []
    yelp_label, imdb_label, sst_label = [], [], []

    for eval in eval_list:
        if eval[0] == '0':
            yelp.append(eval[1])
            yelp_label.append(eval[2])
        elif eval[0] == '1':
            imdb.append(eval[1])
            imdb_label.append(eval[2])
        else:
            sst.append(eval[1])
            sst_label.append(eval[2])

    print('--YELP--')
    print(
        f'ACC: {accuracy_score(yelp, yelp_label)}, F1: {f1_score(yelp, yelp_label)}')
    print('--IMDB--')
    print(
        f'ACC: {accuracy_score(imdb, imdb_label)}, F1: {f1_score(imdb, imdb_label)}')
    print('--SST--')
    print(
        f'ACC: {accuracy_score(sst, sst_label)}, F1: {f1_score(sst, sst_label)}')
    print('--ALL--')
    print(
        f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}, F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}')


def test_self_learning(percent, name):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    time, latency, eval_list = test_sl()

    print('Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    yelp, imdb, sst = [], [], []
    yelp_label, imdb_label, sst_label = [], [], []

    for eval in eval_list:
        if eval[0] == '0':
            yelp.append(eval[1])
            yelp_label.append(eval[2])
        elif eval[0] == '1':
            imdb.append(eval[1])
            imdb_label.append(eval[2])
        else:
            sst.append(eval[1])
            sst_label.append(eval[2])

    print('--YELP--')
    print(
        f'ACC: {accuracy_score(yelp, yelp_label)}, F1: {f1_score(yelp, yelp_label)}')
    print('--IMDB--')
    print(
        f'ACC: {accuracy_score(imdb, imdb_label)}, F1: {f1_score(imdb, imdb_label)}')
    print('--SST--')
    print(
        f'ACC: {accuracy_score(sst, sst_label)}, F1: {f1_score(sst, sst_label)}')
    print('--ALL--')
    print(
        f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}, F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}')


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
# test_bert(percent='0_5', batch_size=64, epochs=10,
#           lr=5e-5, name='data', inf_batch_size=8)


# # 1 %
# test_bert(percent='1', batch_size=64, epochs=10,
#           lr=5e-5, name='data', inf_batch_size=2000)


# ------------------------------------------------------------------------ #

# SELF LEARNING
# 0.5 %
# test_self_learning('0_5', 'data')


# 1 %
test_self_learning('1', 'data')


# ------------------------------------------------------------------------ #


# RANDOM
# 0.5 %
# test_random(percent='0_5', batch_size=10000, name='data')


# 1 %
# test_random(percent='1', batch_size=10000, name='data')
