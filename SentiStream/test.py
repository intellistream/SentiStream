# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config

from main import stream_process, init_train
from kafka_producer import create_stream

from collections import defaultdict


def load_pretrained_models(percent, name='data'):
    """
    Rename pretrained word vector (if best model is saved under 'trained_models') and torch models
    with best weights for current inference. To avoid overwriting while updatingm models.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
    """
    # Word2Vec.load(
    #     f'trained_models/best_{name}_{percent}.model').save(config.SSL_WV)
    # torch.save(torch.load(
    #     f'trained_models/best_{name}_{percent}.pth'), config.SSL_CLF)
    Word2Vec.load(
        f'trained_models/best_{name}.model').save(config.SSL_WV)
    torch.save(torch.load(
        f'trained_models/best_{name}.pth'), config.SSL_CLF)


def test_sentistream(percent, batch_size, lr, test_size, min_count=5, use_pretrained=True,
                     lower_thresh=0.8, update_thresh=20000, update_lex=True,
                     dyn_lex=True, sim_thresh=0.9, dyn_thresh=True, name='data', tag=''):
    """
    Evaluate performance metrics of SentiStream.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
        batch_size (_type_): Batch size for initial training of torch model.
        lr (_type_): Learning rate for initial training of torch model.
        test_size (_type_): Test size for initial training of torch model.
        use_pretrained (bool, optional): Flag to use pretrained best model. Defaults to True.
        lower_thresh (float, optional): Lower threshold for stream merge.
        update_thresh (int, optional): Update threshold for updating models.
    """
    config.DATA = f'data/{name}_{percent}{tag}.csv'
    config.TRAIN_DATA = f'data/{name}_train_{percent}{tag}.csv'

    count = create_stream()

    if use_pretrained:
        load_pretrained_models(percent, name)
    else:
        init_train(batch_size=batch_size, lr=lr,
                   test_size=test_size, min_count=min_count)

    # assert True == False

    time, senti_latency, us_latency, ss_latency, us_eval, ss_eval, senti_eval = stream_process(
        lower_thresh, update_thresh, update_lex, sim_thresh, dyn_lex, dyn_thresh)

    print('SentiStream Latency: ', senti_latency, 'ms')
    print('US Latency: ', us_latency, 'ms')
    print('SS Latency: ', ss_latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    # yelp_us, yelp_ss, yelp_senti = [], [], []
    # imdb_us, imdb_ss, imdb_senti = [], [], []
    # sst_us, sst_ss, sst_senti = [], [], []

    # yelp_us_conf, yelp_ss_conf, yelp_senti_conf = [], [], []
    # imdb_us_conf, imdb_ss_conf, imdb_senti_conf = [], [], []
    # sst_us_conf, sst_ss_conf, sst_senti_conf = [], [], []

    # yelp_label, imdb_label, sst_label = [], [], []

    # senti140_us, senti140_ss, senti140_senti = [], [], []
    # senti140_us_conf, senti140_ss_conf, senti140_senti_conf = [], [], []
    # senti140_label = []

    us, ss, senti = defaultdict(
        list), defaultdict(list), defaultdict(list)
    us_conf, ss_conf, senti_conf = defaultdict(
        list), defaultdict(list), defaultdict(list)
    label = defaultdict(list)

    all_us, all_ss, all_senti = [], [], []
    all_us_conf, all_ss_conf, all_senti_conf = [], [], []
    all_label = []

    for us, ss, senti in zip(us_eval, ss_eval, senti_eval):
        us[us[0]].append(us[1])
        ss[us[0]].append(ss[1])
        senti[us[0]].append(senti[1])

        us_conf[us[0]].append(us[3])
        ss_conf[us[0]].append(ss[3])
        senti_conf[us[0]].append(senti[3])

        label[us[0]].append(us[2])

        all_us.append(us[1])
        all_ss.append(ss[1])
        all_senti.append(senti[1])

        all_us_conf.append(us[3])
        all_ss_conf.append(ss[3])
        all_senti_conf.append(senti[3])

        all_label.append(us[2])

    for key, _ in us.items():
        print(f'\n--{key}--')
        print(
            f'US ACC: {accuracy_score(label[key], us[key])},'
            f' F1: {f1_score(label[key], us[key])}',
            f' AUC: {roc_auc_score(label[key], us_conf[key])}')
        print(
            f'SS ACC: {accuracy_score(label[key], ss[key])},'
            f' F1: {f1_score(label[key], ss[key])}',
            f' AUC: {roc_auc_score(label[key], ss_conf[key])}')
        print(
            f'Senti ACC: {accuracy_score(label[key], senti[key])},'
            f' F1: {f1_score(label[key], senti[key])}',
            f' AUC: {roc_auc_score(label[key], senti_conf[key])}')

    print('\n--ALL--\n')
    print(
        f'US ACC: {accuracy_score(all_label, all_us)},'
        f' F1: {f1_score(all_label, all_us)}',
        f' AUC: {roc_auc_score(all_label, all_us_conf)}')
    print(
        f'SS ACC: {accuracy_score(all_label, all_ss)},'
        f' F1: {f1_score(all_label, all_ss)}',
        f' AUC: {roc_auc_score(all_label, all_ss_conf)}')
    print(
        f'Senti ACC: {accuracy_score(all_label, all_senti)},'
        f' F1: {f1_score(all_label, all_senti)}',
        f' AUC: {roc_auc_score(all_label, all_senti_conf)}')

    # for us, ss, senti in zip(us_eval, ss_eval, senti_eval):
    #     if us[0] == '0':
    #         yelp_us.append(us[1])
    #         yelp_ss.append(ss[1])
    #         yelp_senti.append(senti[1])

    #         yelp_us_conf.append(us[3])
    #         yelp_ss_conf.append(ss[3])
    #         yelp_senti_conf.append(senti[3])

    #         yelp_label.append(us[2])
    #     elif us[0] == '1':
    #         imdb_us.append(us[1])
    #         imdb_ss.append(ss[1])
    #         imdb_senti.append(senti[1])

    #         imdb_us_conf.append(us[3])
    #         imdb_ss_conf.append(ss[3])
    #         imdb_senti_conf.append(senti[3])

    #         imdb_label.append(us[2])
    #     elif us[0] == '2':
    #         sst_us.append(us[1])
    #         sst_ss.append(ss[1])
    #         sst_senti.append(senti[1])

    #         sst_us_conf.append(us[3])
    #         sst_ss_conf.append(ss[3])
    #         sst_senti_conf.append(senti[3])

    #         sst_label.append(us[2])

    #     else:
    #         senti140_us.append(us[1])
    #         senti140_ss.append(ss[1])
    #         senti140_senti.append(senti[1])

    #         senti140_us_conf.append(us[3])
    #         senti140_ss_conf.append(ss[3])
    #         senti140_senti_conf.append(senti[3])

    #         senti140_label.append(us[2])

    # if name == 'data':
    #     print('--YELP--')
    #     print(
    #         f'US ACC: {accuracy_score(yelp_us, yelp_label)}, F1: {f1_score(yelp_us, yelp_label)},',
    #         f' AUC: {roc_auc_score(yelp_label, yelp_us_conf)}')
    #     print(
    #         f'SS ACC: {accuracy_score(yelp_ss, yelp_label)}, F1: {f1_score(yelp_ss, yelp_label)},',
    #         f' AUC: {roc_auc_score(yelp_label, yelp_ss_conf)}')
    #     print(
    #         f'Senti ACC: {accuracy_score(yelp_senti, yelp_label)},'
    #         f' F1: {f1_score(yelp_senti, yelp_label)},',
    #         f' AUC: {roc_auc_score(yelp_label, yelp_senti_conf)}')

    #     print('--IMDB--')
    #     print(
    #         f'US ACC: {accuracy_score(imdb_us, imdb_label)}, F1: {f1_score(imdb_us, imdb_label)},',
    #         f' AUC: {roc_auc_score(imdb_label, imdb_us_conf)}')
    #     print(
    #         f'SS ACC: {accuracy_score(imdb_ss, imdb_label)}, F1: {f1_score(imdb_ss, imdb_label)},',
    #         f' AUC: {roc_auc_score(imdb_label, imdb_ss_conf)}')
    #     print(
    #         f'Senti ACC: {accuracy_score(imdb_senti, imdb_label)},'
    #         f' F1: {f1_score(imdb_senti, imdb_label)},',
    #         f' AUC: {roc_auc_score(imdb_label, imdb_senti_conf)}')

    #     print('--SST--')
    #     print(
    #         f'US ACC: {accuracy_score(sst_us, sst_label)}, F1: {f1_score(sst_us, sst_label)},',
    #         f' AUC: {roc_auc_score(sst_label, sst_us_conf)}')
    #     print(
    #         f'SS ACC: {accuracy_score(sst_ss, sst_label)}, F1: {f1_score(sst_ss, sst_label)},',
    #         f' AUC: {roc_auc_score(sst_label, sst_ss_conf)}')
    #     print(
    #         f'Senti ACC: {accuracy_score(sst_senti, sst_label)}, F1: {f1_score(sst_senti, sst_label)},',
    #         f' AUC: {roc_auc_score(sst_label, sst_senti_conf)}')

    # print('--ALL--')
    # print(
    #     f'US ACC: {accuracy_score(yelp_us+imdb_us+sst_us, yelp_label+imdb_label+sst_label)},'
    #     f' F1: {f1_score(yelp_us+imdb_us+sst_us, yelp_label+imdb_label+sst_label)}',
    #     f' AUC: {roc_auc_score(yelp_label+imdb_label+sst_label,yelp_us_conf+imdb_us_conf+sst_us_conf)}')
    # print(
    #     f'SS ACC: {accuracy_score(yelp_ss+imdb_ss+sst_ss, yelp_label+imdb_label+sst_label)},'
    #     f' F1: {f1_score(yelp_ss+imdb_ss+sst_ss, yelp_label+imdb_label+sst_label)}',
    #     f' AUC: {roc_auc_score(yelp_label+imdb_label+sst_label, yelp_ss_conf+imdb_ss_conf+sst_ss_conf)}')
    # print(
    #     f'Senti ACC: {accuracy_score( yelp_senti+imdb_senti+sst_senti,yelp_label+imdb_label+sst_label)},'
    #     f' F1: {f1_score(yelp_senti+imdb_senti+sst_senti, yelp_label+imdb_label+sst_label)}',
    #     f' AUC: {roc_auc_score(yelp_label+imdb_label+sst_label, yelp_senti_conf+imdb_senti_conf+sst_senti_conf)}')
    # else:
    # print('--ALL--')
    # print(
    #     f'US ACC: {accuracy_score(senti140_label, senti140_us)},'
    #     f' F1: {f1_score(senti140_label, senti140_us)}',
    #     f' AUC: {roc_auc_score(senti140_label, senti140_us_conf)}')
    # print(
    #     f'SS ACC: {accuracy_score(senti140_label, senti140_ss)},'
    #     f' F1: {f1_score(senti140_label, senti140_ss)}',
    #     f' AUC: {roc_auc_score(senti140_label, senti140_ss_conf)}')
    # print(
    #     f'Senti ACC: {accuracy_score(senti140_label, senti140_senti)},'
    #     f' F1: {f1_score(senti140_label, senti140_senti)}',
    #     f' AUC: {roc_auc_score(senti140_label, senti140_senti_conf)}')


# # ----------------------------------------------------------------------------------------------- #


# # 0.5 %
# print('\n0.5% => Yelp -> IMDb -> SST-2 \n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=20000, tag='_percent')

test_sentistream(percent='1', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
                 use_pretrained=True, lower_thresh=0.8, update_thresh=20000, tag='_percent')

# # ----------------------------------------------------------------------------------------------- #
# # Additional tests
# # ----------------------------------------------------------------------------------------------- #

# # 1 %
# print('\n1% => Yelp -> IMDb -> SST-2 \n')
# test_sentistream(percent='1', batch_size=256, lr=0.005, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.7, update_thresh=20000, tag='_percent')

# # 10 %
# print('\n10% => Yelp -> IMDb -> SST-2 \n')
# test_sentistream(percent='10', batch_size=256, lr=0.001, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=20000, tag='_percent')

# # Amazon
# print('\n Trained on 1 Quarter - Amazon 0.8 \n')
# test_sentistream(percent='1', batch_size=64, lr=0.008, test_size=0.2, min_count=5,  # 0.005 -> 73
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=10000, name='amazon')

# # ----------------------------------------------------------------------------------------------- #


# # IMDb -> SST -> YELP

# # 0.5 %
# print('\n0.5% => IMDb -> SST-2 -> Yelp \n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=20000, name='data_isy', tag='_percent')

# # SST -> YELP -> IMDb

# # 0.5 %
# print('\n0.5% => SST-2 -> Yelp -> IMDb \n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=20000, name='data_syi', tag='_percent')


# # ----------------------------------------------------------------------------------------------- #

# # components test
# # #  baseline
# print('\nComponents test\n')
# print('\nBaseline\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=False, tag='_percent')

# # # dyn lex update
# # # # 0.7
# print('\nDyn lex update - similarity threshod - 0.7\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.7, tag='_percent')

# # # # 0.8
# print('\nDyn lex update - similarity threshod - 0.8\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.8, tag='_percent')

# # # # 0.9
# print('\nDyn lex update - similarity threshod - 0.9\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.9, tag='_percent')

# # # dyn threshold
# # # # 0.7
# print('\nDyn threshold - lower threshod - 0.7\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.7,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True, tag='_percent')

# # # # 0.8
# print('\nDyn threshold - lower threshod - 0.8\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True, tag='_percent')

# # # # 0.9
# print('\nDyn threshold - lower threshod - 0.9\n')
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.9,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True, tag='_percent')

# ----------------------------------------------------------------------------------------------- #
# Additional tests - Temporal dataset
# ----------------------------------------------------------------------------------------------- #

# 1 day
# print('\n Trained on 1 day - Senti140 \n')
# for _ in range(2):
#     test_sentistream(percent='1', batch_size=64, lr=0.008, test_size=0.2, min_count=5,
#                      use_pretrained=True, lower_thresh=0.8, update_thresh=40000, name='senti140')

# # 2 day
# print('\n Trained on 2 days - Senti140 \n')
# test_sentistream(percent='2', batch_size=64, lr=0.008, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.8, update_thresh=40000, name='senti140')