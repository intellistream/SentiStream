# pylint: disable=import-error
import os
import pandas as pd


def generate_df(df, dataset):
    """
    Generate train and inf dataset from data.

    Args:
        df (DataFrame): DataFrame to be saved.
        dataset (str): Name of dataset.
    """
    p_1 = int(len(df) * 0.01)
    p_0_5 = p_1 // 2

    if not os.path.exists('data'):
        os.makedirs('data')

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


yelp_df = pd.read_csv('yelp.csv', header=None)
yelp_df.insert(0, 'id', 0)
imdb_df = pd.read_csv('imdb.csv', header=None)
imdb_df.insert(0, 'id', 1)
sst_df = pd.read_csv('sst.csv', header=None)
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
