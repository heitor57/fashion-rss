from utils import create_path_to_file
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
    product_ids = np.unique(
        np.hstack((train_df.product_id.unique(), test_df.product_id.unique())))
    num_users = len(user_ids)
    num_products = len(product_ids)

    user_int_ids = integer_map(user_ids)
    product_int_ids = integer_map(product_ids)

    train_df.user_id = train_df.user_id.map(lambda x: user_int_ids[x])
    train_df.product_id = train_df.product_id.map(lambda x: product_int_ids[x])

    test_df.user_id = test_df.user_id.map(lambda x: user_int_ids[x])
    test_df.product_id = test_df.product_id.map(lambda x: product_int_ids[x])

    train_normalized_df = train_df.groupby(['user_id', 'product_id'
                                           ])['is_click'].sum().reset_index()

    train_normalized_df = train_normalized_df[[
        'user_id', 'product_id', 'is_click'
    ]]

    test_normalized_df = test_df[['user_id', 'product_id','query_id']].copy()

    def _f(x):
        if x in product_int_ids:
            return product_int_ids[x]
        else:
            return x

    attributes_df.product_id = attributes_df.product_id.map(_f)

    return train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids


def negative_samples( df):
    df = df.set_index('user_id')
    print(df)
    df['products_sampled'] = 0
    products_id = set(df.product_id.unique())
    users_products = df.groupby('user_id')['product_id'].unique().to_dict()
    users_products = {k:set(v) for k,v in users_products.items()}
    # print(users_products)
# set(users_products.loc[user_id])
    for user_id in tqdm(df.index):
        user_df = df.loc[[user_id]]
        # print(user_df)
        products_sample_space = np.array(list(products_id - users_products[user_id]))
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


def parquet_save(df,file_name):
    create_path_to_file(file_name)
    df.to_parquet(open(file_name,'wb'))
