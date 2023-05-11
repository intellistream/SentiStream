import csv

import config

from kafka_producer import create_stream
from other_exp.bert_training import train
from other_exp.bert_inf import get_results


def test_bert(batch_size, epochs, lr, name, inf_batch_size):
    config.DATA = f'data/{name}_0_5_percent.csv'
    config.TRAIN_DATA = f'data/{name}_train_0_5_percent.csv'

    count = create_stream()

    # comment this if model is already trained
    # train(batch_size, epochs, lr, name)

    time, latency, acc, f1 = get_results(name, inf_batch_size)

    print('Avg Latency: ', latency, 'ms')
    print('Elapsed time: ', time, 's')
    print(count / time, 'tuples per sec')

    print('Avg US_ACC: ', sum(acc)/len(acc))
    print('Avg US_F1: ', sum(f1)/len(f1))

    with open(f'outputs/bert_output_{name}.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['us_acc', 'us_f1', 'ss_acc',
                        'ss_f1', 'senti_acc', 'senti_f1'])

        for row in zip(*[acc, f1]):
            writer.writerow(row)


# # 0.5 %
# combined
print('\n--Combined Dataset--\n')
test_bert(batch_size=64, epochs=5, lr=5e-5, name='data', inf_batch_size=8)

# # yelp
# print('\n--Yelp Dataset--\n')
# test_bert(batch_size=32, epochs=10, lr=1e-6, name='yelp', inf_batch_size=8)

# # imdb
# print('\n--IMDb Dataset--\n')
# test_bert(batch_size=64, epochs=5, lr=5e-5, name='imdb', inf_batch_size=8)

# # sst-2
# print('\n--SST-2 Dataset--\n')
# test_bert(batch_size=64, epochs=3, lr=5e-5, name='sst', inf_batch_size=8)
