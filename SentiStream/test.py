# pylint: disable=import-error
# pylint: disable=no-name-in-module
import torch

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score

import config

from main import stream_process, init_train
from kafka_producer import create_stream


def load_pretrained_models(percent):
    """
    Rename pretrained word vector (if best model is saved under 'trained_models') and torch models
    with best weights for current inference. To avoid overwriting while updatingm models.

    Args:
        percent (str): Flag to distinguish models trained on different percentages of data.
    """
    Word2Vec.load(
        f'trained_models/best_data_{percent}.model').save(config.SSL_WV)
    torch.save(torch.load(
        f'trained_models/best_data_{percent}.pth'), config.SSL_CLF)


def test_sentistream(percent, batch_size, lr, test_size, min_count=5, use_pretrained=True,
                     lower_thresh=0.8, update_thresh=20000, update_lex=True,
                     dyn_lex=True, sim_thresh=0.9, dyn_thresh=True):
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
    config.DATA = f'data/data_{percent}_percent.csv'
    config.TRAIN_DATA = f'data/data_train_{percent}_percent.csv'

    count = create_stream()

    if use_pretrained:
        load_pretrained_models(percent)
    else:
        init_train(batch_size=batch_size, lr=lr,
                   test_size=test_size, min_count=min_count)

    time, senti_latency, us_latency, ss_latency, us_eval, ss_eval, senti_eval = stream_process(
        lower_thresh, update_thresh, update_lex, sim_thresh, dyn_lex, dyn_thresh)
    print('SentiStream Latency: ', senti_latency, 'ms')
    print('US Latency: ', us_latency, 'ms')
    print('SS Latency: ', ss_latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    yelp_us, yelp_ss, yelp_senti = [], [], []
    imdb_us, imdb_ss, imdb_senti = [], [], []
    sst_us, sst_ss, sst_senti = [], [], []
    yelp_label, imdb_label, sst_label = [], [], []

    for us, ss, senti in zip(us_eval, ss_eval, senti_eval):
        if us[0] == '0':
            yelp_us.append(us[1])
            yelp_ss.append(ss[1])
            yelp_senti.append(senti[1])
            yelp_label.append(us[2])
        elif us[0] == '1':
            imdb_us.append(us[1])
            imdb_ss.append(ss[1])
            imdb_senti.append(senti[1])
            imdb_label.append(us[2])
        else:
            sst_us.append(us[1])
            sst_ss.append(ss[1])
            sst_senti.append(senti[1])
            sst_label.append(us[2])

    print('--YELP--')
    print(
        f'US ACC: {accuracy_score(yelp_us, yelp_label)}, F1: {f1_score(yelp_us, yelp_label)}')
    print(
        f'SS ACC: {accuracy_score(yelp_ss, yelp_label)}, F1: {f1_score(yelp_ss, yelp_label)}')
    print(
        f'Senti ACC: {accuracy_score(yelp_senti, yelp_label)}, F1: {f1_score(yelp_senti, yelp_label)}')

    print('--IMDB--')
    print(
        f'US ACC: {accuracy_score(imdb_us, imdb_label)}, F1: {f1_score(imdb_us, imdb_label)}')
    print(
        f'SS ACC: {accuracy_score(imdb_ss, imdb_label)}, F1: {f1_score(imdb_ss, imdb_label)}')
    print(
        f'Senti ACC: {accuracy_score(imdb_senti, imdb_label)}, F1: {f1_score(imdb_senti, imdb_label)}')

    print('--SST--')
    print(
        f'US ACC: {accuracy_score(sst_us, sst_label)}, F1: {f1_score(sst_us, sst_label)}')
    print(
        f'SS ACC: {accuracy_score(sst_ss, sst_label)}, F1: {f1_score(sst_ss, sst_label)}')
    print(
        f'Senti ACC: {accuracy_score(sst_senti, sst_label)}, F1: {f1_score(sst_senti, sst_label)}')

    print('--ALL--')
    print(
        f'US ACC: {accuracy_score(yelp_us+imdb_us+sst_us, yelp_label+imdb_label+sst_label)}, F1: {f1_score(yelp_us+imdb_us+sst_us, yelp_label+imdb_label+sst_label)}')
    print(
        f'SS ACC: {accuracy_score(yelp_ss+imdb_ss+sst_ss, yelp_label+imdb_label+sst_label)}, F1: {f1_score(yelp_ss+imdb_ss+sst_ss, yelp_label+imdb_label+sst_label)}')
    print(
        f'Senti ACC: {accuracy_score(yelp_senti+imdb_senti+sst_senti, yelp_label+imdb_label+sst_label)}, F1: {f1_score(yelp_senti+imdb_senti+sst_senti, yelp_label+imdb_label+sst_label)}')

# ----------------------------------------------------------------------------------------------- #


# 0.5 %
test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2, min_count=5,
                 use_pretrained=True, lower_thresh=0.8, update_thresh=20000)


# ----------------------------------------------------------------------------------------------- #

# 1 %
# test_sentistream(percent='1', batch_size=256, lr=0.005, test_size=0.2, min_count=5,
#                  use_pretrained=True, lower_thresh=0.7, update_thresh=20000)


# ----------------------------------------------------------------------------------------------- #

# components test
# #  baseline
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=False)

# # dyn lex update
# # # 0.7
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.7)

# # # 0.8
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.8)

# # # 0.9
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=True, dyn_thresh=False, sim_thresh=0.9)

# # dyn threshold
# # # 0.7
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.7,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True)

# # # 0.8
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.8,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True)

# # # 0.9
# test_sentistream(percent='0_5', batch_size=64, lr=0.0008, test_size=0.2,
#                  min_count=5, use_pretrained=True, lower_thresh=0.9,
#                  update_thresh=20000, dyn_lex=False, dyn_thresh=True)
