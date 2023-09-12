# pylint: disable=import-error
import os
import pandas as pd

if not os.path.exists('data'):
    os.makedirs('data')


def generate_df(df, dataset):
    """
    Generate train and inf dataset from data.

    Args:
        df (DataFrame): DataFrame to be saved.
        dataset (str): Name of dataset.
    """
    p_1 = int(len(df) * 0.01)
    p_0_5 = p_1 // 2

    # 1%
    df.iloc[:p_1, :].to_csv(f'data/{dataset}_train_1_percent.csv',
                            header=False, index=False)
    df.iloc[p_1:, :].to_csv(
        f'data/{dataset}_1_percent.csv', header=False, index=False)

    # 0.5%
    df.iloc[:p_0_5, :].to_csv(
        f'data/{dataset}_train_0_5_percent.csv', header=False, index=False)
    df.iloc[p_0_5:, :].to_csv(f'data/{dataset}_0_5_percent.csv',
                              header=False, index=False)


def generate_temporal_df(df, dataset):
    """
    Generate temporal train and inf dataset from data.

    Args:
        df (DataFrame): DataFrame to be saved.
        dataset (str): Name of dataset.
    """

    df_1 = df[df.iloc[:, -1] == 1].iloc[:, :-1]
    df_2 = df[df.iloc[:, -1].isin([1, 11])].iloc[:, :-1]

    # 2 days
    df_2.to_csv(f'data/{dataset}_train_2.csv',
                header=False, index=False)

    df.drop(df_2.index).to_csv(
        f'data/{dataset}_2.csv', header=False, index=False)

    # 1 day
    df_1.to_csv(f'data/{dataset}_train_1.csv',
                header=False, index=False)
    df.drop(df_1.index).to_csv(
        f'data/{dataset}_1.csv', header=False, index=False)


def generate_temporal_df2(df, dataset):
    """
    Generate temporal train and inf dataset from data.

    Args:
        df (DataFrame): DataFrame to be saved.
        dataset (str): Name of dataset.
    """

    df_1 = df[df.iloc[:, 0] == '2010Q1']

    print(len(df_1), len(df.drop(df_1.index)))

    # 1 Q
    df_1.to_csv(f'data/{dataset}_train_1.csv',
                header=False, index=False)
    df.drop(df_1.index).to_csv(
        f'data/{dataset}_1.csv', header=False, index=False)


yelp_df = pd.read_csv('../../../Downloads/senti/yelp.csv', header=None)
yelp_df.insert(0, 'id', 0)
imdb_df = pd.read_csv('../../../Downloads/senti/imdb.csv', header=None)
imdb_df.insert(0, 'id', 1)
sst_df = pd.read_csv('../../../Downloads/senti/sst.csv', header=None)
sst_df.insert(0, 'id', 2)

merged_df = pd.concat([yelp_df, imdb_df, sst_df])
merged_df.iloc[-1, 0] = -1

generate_df(merged_df, 'data')

merged_df = pd.concat([imdb_df, sst_df, yelp_df])
merged_df.iloc[-1, 0] = -1

generate_df(merged_df, 'data_isy')

merged_df = pd.concat([sst_df, yelp_df, imdb_df])
merged_df.iloc[-1, 0] = -1
generate_df(merged_df, 'data_syi')

senti140 = pd.read_csv('../../../Downloads/senti/senti140.csv', header=None)
senti140.insert(0, 'id', 3)
generate_temporal_df(senti140, 'senti140')


amazon = pd.read_csv('../../../Downloads/senti/amazon.csv', header=None)
amazon = amazon.rename(columns={0: 'id'})
generate_temporal_df2(amazon, 'amazon')

print('yelp', yelp_df.iloc[:, 1].value_counts())
print('imdb', imdb_df.iloc[:, 1].value_counts())
print('sst', sst_df.iloc[:, 1].value_counts())
print('senti140', senti140.iloc[:, 1].value_counts())
print('amazon', amazon.iloc[:, 1].value_counts())


#               NEG     POS

#   Yelp        40227   39773
#   LMRD        24698   24884
#   SST         30076   37779
#   Senti140    800000  800000
#   Amazon      170924  610225
