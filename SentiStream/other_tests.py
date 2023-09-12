# pylint: disable=import-error
# pylint: disable=no-name-in-module
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from gensim.models import Word2Vec
import config

from kafka_producer import create_stream
from other_exp.bert_training import train
from other_exp.bert_inf import get_results
from other_exp.random_pred import test
from other_exp.self_learning import test_sl
from other_exp.ann.trainer import Trainer
from other_exp.ann.inf import get_preds

from collections import defaultdict

from main import stream_process, init_train


def test_supervised_bert(percent, batch_size, epochs, lr, inf_batch_size, name='data', tag=''):
    config.DATA = f'data/{name}_{percent}{tag}.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}{tag}.csv'

    count = create_stream()

    # comment this if model is already trained
    # train(batch_size, epochs, lr, f'bert_{name}_' + percent)

    time, latency, eval_list = get_results(
        f'bert_{name}_' + percent, inf_batch_size)

    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    pred, conf, label = defaultdict(
        list), defaultdict(list), defaultdict(list)

    all, all_conf, all_label = [], [], []

    for eval in eval_list:
        pred[eval[0]].append(eval[1])
        label[eval[0]].append(eval[2])
        conf[eval[0]].append(eval[3])

        all.append(eval[1])
        all_label.append(eval[2])
        all_conf.append(eval[3])

    for key, _ in pred.items():
        print(f'\n--{key}--')
        print(
            f'ACC: {accuracy_score(label[key], pred[key])},'
            f' F1: {f1_score(label[key], pred[key])}',
            f' AUC: {roc_auc_score(label[key], conf[key])}')

    print('\n--ALL--\n')
    print(
        f'ACC: {accuracy_score(all_label, all)},'
        f' F1: {f1_score(all_label, all)}',
        f' AUC: {roc_auc_score(all_label, all_conf)}')


def test_supervised_w2v(percent, batch_size, epochs, lr, name='data', tag=''):
    w2v = Word2Vec.load(
        f'trained_models/best_{name}_{percent}.model')

    config.DATA = f'data/{name}_{percent}{tag}.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}{tag}.csv'

    count = create_stream()

    model = Trainer(wv_model=w2v, batch_size=batch_size,
                    learning_rate=lr).fit_and_save(epochs=epochs)

    time, latency, eval_list = get_preds(model, w2v, 10000)

    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    yelp, imdb, sst = [], [], []
    yelp_label, imdb_label, sst_label = [], [], []
    yelp_conf, imdb_conf, sst_conf = [], [], []

    senti140, senti140_conf, senti140_label = [], [], []

    for eval in eval_list:
        if eval[0] == '0':
            yelp.append(eval[1])
            yelp_label.append(eval[2])
            yelp_conf.append(eval[3])
        elif eval[0] == '1':
            imdb.append(eval[1])
            imdb_label.append(eval[2])
            imdb_conf.append(eval[3])
        elif eval[0] == '2':
            sst.append(eval[1])
            sst_label.append(eval[2])
            sst_conf.append(eval[3])
        else:
            senti140.append(eval[1])
            senti140_label.append(eval[2])
            senti140_conf.append(eval[3])

    if name == 'data':

        print('--YELP--')
        print(
            f'ACC: {accuracy_score(yelp, yelp_label)}, F1: {f1_score(yelp, yelp_label)},'
            f' AUC: {roc_auc_score(yelp_label, yelp_conf)}')
        print('--IMDB--')
        print(
            f'ACC: {accuracy_score(imdb, imdb_label)}, F1: {f1_score(imdb, imdb_label)},'
            f' AUC: {roc_auc_score(imdb_label, imdb_conf)}')
        print('--SST--')
        print(
            f'ACC: {accuracy_score(sst, sst_label)}, F1: {f1_score(sst, sst_label)},'
            f' AUC: {roc_auc_score(sst_label, sst_conf)}')
        print('--ALL--')
        print(
            f'ACC: {accuracy_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)},'
            f' F1: {f1_score(yelp+imdb+sst, yelp_label+imdb_label+sst_label)},'
            f' AUC: {roc_auc_score(yelp_label+imdb_label+sst_label, yelp_conf+imdb_conf+sst_conf)}')
    else:

        print('--Senti140--')
        print(
            f'ACC: {accuracy_score(senti140, senti140_label)}, F1: {f1_score(senti140, senti140_label)},'
            f' AUC: {roc_auc_score(senti140_label, senti140_conf)}')


def test_self_learning(percent, name='data', tag=''):
    config.DATA = f'data/{name}_{percent}{tag}.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}{tag}.csv'

    count = create_stream()

    time, latency, eval_list = test_sl()

    print('Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    pred, conf, label = defaultdict(
        list), defaultdict(list), defaultdict(list)

    all, all_conf, all_label = [], [], []

    for eval in eval_list:
        pred[eval[0]].append(eval[1])
        label[eval[0]].append(eval[2])
        conf[eval[0]].append(eval[3])

        all.append(eval[1])
        all_label.append(eval[2])
        all_conf.append(eval[3])

    for key, _ in pred.items():
        print(f'\n--{key}--')
        print(
            f'ACC: {accuracy_score(label[key], pred[key])},'
            f' F1: {f1_score(label[key], pred[key])}',
            f' AUC: {roc_auc_score(label[key], conf[key])}')

    print('\n--ALL--\n')
    print(
        f'ACC: {accuracy_score(all_label, all)},'
        f' F1: {f1_score(all_label, all)}',
        f' AUC: {roc_auc_score(all_label, all_conf)}')


def test_random(percent, batch_size, name='data', tag=''):
    config.DATA = f'data/{name}_{percent}{tag}.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}{tag}.csv'

    create_stream()

    acc, f1, auc = test(batch_size)

    print('Avg ACC: ', sum(acc)/len(acc))
    print('Avg F1: ', sum(f1)/len(f1))
    print('Avg AUC: ', auc)


# # # BERT
# # # 0.5 %
# print('\nSupervised BERT\n')
# print('\n0.5% => Yelp -> IMDb -> SST-2  \n')
# test_supervised_bert(percent='0_5', batch_size=64,
#                      epochs=10, lr=5e-5, inf_batch_size=8, tag='_percent')

# print('\n0.5% => IMDb -> SST-2 -> Yelp \n')
# test_supervised_bert(percent='0_5', batch_size=64,
#                      epochs=10, lr=5e-5, inf_batch_size=8, name='data_isy', tag='_percent')

# print('\n0.5% => SST-2 -> Yelp -> IMDb \n')
# test_supervised_bert(percent='0_5', batch_size=64,
#                      epochs=10, lr=5e-5, inf_batch_size=8, name='data_syi', tag='_percent')


# # 1 %
# print('\n1% => Yelp -> IMDb -> SST-2 \n')
# test_supervised_bert(percent='1', batch_size=64, epochs=10,
#                      lr=5e-5, inf_batch_size=8, tag='_percent')

# # 10 %
# print('\n10% => Yelp -> IMDb -> SST-2 \n')
# test_supervised_bert(percent='10', batch_size=64, epochs=1,
#                      lr=5e-5, inf_batch_size=8, tag='_percent')


# ------------------------------------------------------------------------ #


# # W2V
# # 0.5 %
# print('\nSupervised W2V\n')
# print('\n0.5% => Yelp -> IMDb -> SST-2  \n')
# test_supervised_w2v('0_5', 256, 100, 3e-3, tag='_percent')

# print('\n0.5% => IMDb -> SST-2 -> Yelp \n')
# test_supervised_w2v('0_5', 256, 100, 3e-3, name='data_isy', tag='_percent')

# print('\n0.5% => SST-2 -> Yelp -> IMDb \n')
# test_supervised_w2v('0_5', 256, 100, 3e-3, name='data_syi', tag='_percent')


# # 1 %
# print('\n1% => Yelp -> IMDb -> SST-2 \n')
# test_supervised_w2v('1', 256, 100, 3e-3, tag='_percent')


# ------------------------------------------------------------------------ #


# SELF LEARNING
# 0.5 %
# print('\nSelf-Learning\n')
# print('\n0.5% => Yelp -> IMDb -> SST-2  \n')
# test_self_learning('0_5', tag='_percent')

# print('\n0.5% => IMDb -> SST-2 -> Yelp \n')
# test_self_learning('0_5', name='data_isy', tag='_percent')

# print('\n0.5% => SST-2 -> Yelp -> IMDb \n')
# test_self_learning('0_5', name='data_syi', tag='_percent')


# # 1 %
# print('\n1% => Yelp -> IMDb -> SST-2 \n', tag='_percent')
# test_self_learning('1')


# ------------------------------------------------------------------------ #


# # RANDOM
# # 0.5 %
# print('\nRandom\n')
# print('\n0.5% => Yelp -> IMDb -> SST-2  \n')
# test_random(percent='0_5', batch_size=10000, tag='_percent')


# # 1 %
# print('\n1% => Yelp -> IMDb -> SST-2 \n')
# test_random(percent='1', batch_size=10000, tag='_percent')


# ----------------------------------------------------------------------------------------------- #
# Additional tests - Temporal dataset
# ----------------------------------------------------------------------------------------------- #

# # BERT
# print('\nSupervised BERT\n')
# # 1 day
# print('\n Trained on 1 day - Senti140 \n')
# test_supervised_bert(percent='1', batch_size=64,
#                      epochs=1, lr=5e-3, inf_batch_size=8, name='senti140')

# # 2 day
# print('\n Trained on 2 day - Senti140 \n')
# test_supervised_bert(percent='2', batch_size=64,
#                      epochs=1, lr=5e-3, inf_batch_size=8, name='senti140')


# # 1 Q
# print('\n Trained on 1 Quarter - Amazon \n')
# test_supervised_bert(percent='1', batch_size=64,
#                      epochs=2, lr=5e-3, inf_batch_size=8, name='amazon')


# W2V
# print('\nSupervised W2V\n')
# # 1 day
# print('\n Trained on 1 day - Senti140 \n')
# test_supervised_w2v('1', 256, 100, 3e-3, name='senti140')

# # 2 day
# print('\n Trained on 2 day - Senti140 \n')
# test_supervised_w2v('2', 256, 100, 3e-3, name='senti140')


# SELF LEARNING
# print('\nSelf-Learning\n')
# # 1 day
# print('\n Trained on 1 day - Senti140 \n')
# test_self_learning('1', name='senti140')

# # 2 day
# print('\n Trained on 2 day - Senti140 \n')
# test_self_learning('2', name='senti140')

# # 1 Q
# print('\n Trained on 1 Quarter - Amazon \n')
# test_self_learning('1', name='amazon')


# # RANDOM
# print('\nRandom\n')
# # 1 day
# print('\n Trained on 1 day - Senti140 \n')
# test_random(percent='1', batch_size=10000, name='senti140')

# # 2 day
# print('\n Trained on 2 day - Senti140 \n')
# test_random(percent='2', batch_size=10000, name='senti140')

# # 1 Q
# print('\n Trained on 1 Quarter - Amazon \n')
# test_random(percent='1', batch_size=10000, name='amazon')
