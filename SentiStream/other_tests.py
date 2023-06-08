# pylint: disable=import-error
# pylint: disable=no-name-in-module
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec
import config

from kafka_producer import create_stream
from other_exp.bert_training import train
from other_exp.bert_inf import get_results
from other_exp.random_pred import test
from other_exp.self_learning import test_sl
from other_exp.ann.trainer import Trainer
from other_exp.ann.inf import get_preds


def test_supervised_bert(percent, batch_size, epochs, lr, inf_batch_size, name='data'):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    # comment this if model is already trained
    # train(batch_size, epochs, lr, f'bert_{name}_' + percent)

    time, latency, eval_list = get_results(
        f'bert_{name}_' + percent, inf_batch_size)

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
        f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)},'
        f' F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}')


def test_supervised_w2v(percent, batch_size, epochs, lr, name='data'):
    w2v = Word2Vec.load(
        f'trained_models/best_{name}_{percent}.model')

    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    count = create_stream()

    model = Trainer(wv_model=w2v, batch_size=batch_size,
                    learning_rate=lr).fit_and_save(epochs=epochs)

    time, latency, eval_list = get_preds(model, w2v, 10000)

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
        f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)},'
        f' F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}')


def test_self_learning(percent, name='data'):
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
        f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)},'
        f' F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)}')


def test_random(percent, batch_size, name):
    config.DATA = f'data/{name}_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}_percent.csv'

    create_stream()

    acc, f1 = test(batch_size)

    print('Avg ACC: ', sum(acc)/len(acc))
    print('Avg F1: ', sum(f1)/len(f1))


# BERT
# # 0.5 %
# test_supervised_bert(percent='0_5', batch_size=64,
#                      epochs=10, lr=5e-5, inf_batch_size=8)

# test_supervised_bert(percent='0_5', batch_size=64,
    #  epochs=10, lr=5e-5, inf_batch_size=2000, name='data_isy')

# test_supervised_bert(percent='0_5', batch_size=64,
#                      epochs=10, lr=5e-5, inf_batch_size=2000, name='data_syi')


# # 1 %
# test_supervised_bert(percent='1', batch_size=64, epochs=10,
#           lr=5e-5, inf_batch_size=8)


# ------------------------------------------------------------------------ #


# W2V
# 0.5 %
# test_supervised_w2v('0_5', 256, 100, 3e-3)

# test_supervised_w2v('0_5', 256, 100, 3e-3, name='data_isy')

# test_supervised_w2v('0_5', 256, 100, 3e-3, name='data_syi')


# 1 %
# test_supervised_w2v('1', 256, 100, 3e-3)


# ------------------------------------------------------------------------ #


# SELF LEARNING
# 0.5 %
# test_self_learning('0_5')

# test_self_learning('0_5', name='data_isy')

# test_self_learning('0_5', name='data_syi')


# 1 %
# test_self_learning('1')


# ------------------------------------------------------------------------ #


# RANDOM
# 0.5 %
# test_random(percent='0_5', batch_size=10000)


# 1 %
# test_random(percent='1', batch_size=10000)
