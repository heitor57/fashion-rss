from utils import create_path_to_file
import constants
import os.path
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os
import re
import numpy as np
import os
from copy import copy

# class Dataset:


def parquet_load(file_name):
    df = pd.read_parquet(file_name)
    return df


def farfetch_train_test_normalization(train_df, test_df, attributes_df):

    def integer_map(values):
        d = dict()
        for i, v in enumerate(values):
            d[v] = i
        return d

    user_ids = np.unique(
        np.hstack((train_df.user_id.unique(), test_df.user_id.unique())))
    # product_ids = np.unique(
    # np.hstack((train_df.product_id.unique(), test_df.product_id.unique())))
    product_ids = attributes_df.product_id.unique()
    # num_users = len(user_ids)
    # num_products = len(product_ids)

    user_int_ids = integer_map(user_ids)
    product_int_ids = integer_map(product_ids)

    train_df.user_id = train_df.user_id.map(lambda x: user_int_ids[x])
    train_df.product_id = train_df.product_id.map(lambda x: product_int_ids[x])

    test_df.user_id = test_df.user_id.map(lambda x: user_int_ids[x])
    test_df.product_id = test_df.product_id.map(lambda x: product_int_ids[x])

    # train_normalized_df = train_df.groupby(['user_id', 'product_id'
    # ])['is_click'].sum().reset_index()

    # train_normalized_df = train_normalized_df[[
    # 'user_id', 'product_id', 'is_click'
    # ]]

    # test_normalized_df = test_df[['user_id', 'product_id', 'query_id']].copy()

    def _f(x):
        if x in product_int_ids:
            return product_int_ids[x]
        else:
            return x

    attributes_df.product_id = attributes_df.product_id.map(_f)
    attributes_df = attributes_df.sort_values('product_id')
    # train_normalized_df = train_df.loc[
    # train_df['is_click'] > 0]
    columns_to_dummies = [
        'week', 'week_day', 'device_category', 'device_platform', 'user_tier',
        'user_country'
    ]
    for column in columns_to_dummies:
        train_df = pd.concat(
            [train_df,
             pd.get_dummies(train_df[column], prefix=column)], axis=1)
        del train_df[column]
        test_df = pd.concat(
            [test_df, pd.get_dummies(test_df[column], prefix=column)], axis=1)
        del test_df[column]
    columns_to_dummies = [
        'season', 'collection', 'category_id_l1', 'category_id_l2',
        'category_id_l3', 'brand_id', 'season_year'
    ]
    for column in columns_to_dummies:
        attributes_df = pd.concat([
            attributes_df,
            pd.get_dummies(attributes_df[column], prefix=column)
        ],
                                  axis=1)
        del attributes_df[column]
    return train_df, test_df, attributes_df, user_int_ids, product_int_ids


def dataset_accumulate_clicks(df):
    df = df.groupby(['user_id', 'product_id'])['is_click'].sum().reset_index()
    return df


def dataset_filter_zero_clicks(df):
    df = df.loc[df['is_click'] > 0]
    return df


def negative_samples(df):
    df = df.set_index('user_id')
    print(df)
    df['products_sampled'] = 0
    products_id = set(df.product_id.unique())
    users_products = df.groupby('user_id')['product_id'].unique().to_dict()
    users_products = {k: set(v) for k, v in users_products.items()}
    # print(users_products)
    # set(users_products.loc[user_id])
    for user_id in tqdm(df.index):
        user_df = df.loc[[user_id]]
        # print(user_df)
        products_sample_space = np.array(
            list(products_id - users_products[user_id]))
        # i = random.randint(0,len(products_sample_space)-1)
        # user_df

        products_sampled = np.random.choice(products_sample_space,
                                            size=len(user_df),
                                            replace=False)
        df.loc[user_id]['products_sampled'] = products_sampled
        # sampled_product_id = products_sample_space[i]
    return df.reset_index()


def sample_df(df, rate, seed=0):
    df = df.sample(int(len(df) * rate), random_state=seed)
    return df


def parquet_save(df, file_name):
    create_path_to_file(file_name)
    df.to_parquet(open(file_name, 'wb'))


def one_split(df, train_rate):
    train_size = int(len(df) * train_rate)
    test_size = len(df) - train_size

    query_df = df.groupby(['user_id'])['query_id'].unique()

    rng = np.random.default_rng(constants.RANDOM_SEED)
    users_id = df['user_id'].unique()
    rng.shuffle(users_id)
    df = df.set_index('user_id')
    test_queries_id = []
    test_current_size = 0
    for user_id in tqdm(users_id):
        queries_id = set(query_df.loc[user_id])
        query_id = random.sample(queries_id, 1)[0]
        test_queries_id.append(query_id)
        test_current_size += 6
        if test_current_size >= test_size:
            break

    df = df.reset_index()
    test_queries_id = set(test_queries_id)
    condition = df['query_id'].isin(test_queries_id)
    train_df = df.loc[~condition]
    test_df = df.loc[condition]
    # train_size = int(train_rate*len(df))
    # test_size = len(df)-train_size
    print(train_df.shape)
    print(test_df.shape)
    return train_df, test_df
