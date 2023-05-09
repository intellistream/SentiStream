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
    p_0_1 = p_1 // 10
    p_0_5 = p_0_1 * 5

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

    # 0.1%
    df.iloc[:p_0_1, :].to_csv(
        f'data/{dataset}_train_0_1_percent.csv', header=False, index=False)
    df.iloc[p_0_1:, :].to_csv(f'data/{dataset}_0_1_percent.csv',
                              header=False, index=False)


yelp_df = pd.read_csv('yelp.csv', header=None)
imdb_df = pd.read_csv('imdb.csv', header=None)
sst_df = pd.read_csv('sst.csv', header=None)
merged_df = pd.concat([yelp_df, imdb_df, sst_df])


generate_df(merged_df, 'data')
generate_df(yelp_df, 'yelp')
generate_df(imdb_df, 'imdb')
generate_df(sst_df, 'sst')
