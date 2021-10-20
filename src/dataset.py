from utils import create_path_to_file

import json
import gzip
import json
import sklearn.decomposition
import sklearn.feature_selection
import pickle
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

    query_ids = np.unique(
        np.hstack((train_df.query_id.unique(), test_df.query_id.unique())))

    query_int_ids = integer_map(query_ids)

    train_df.user_id = train_df.user_id.map(lambda x: user_int_ids[x])
    train_df.product_id = train_df.product_id.map(lambda x: product_int_ids[x])
    train_df.query_id = train_df.query_id.map(lambda x: query_int_ids[x])

    test_df.user_id = test_df.user_id.map(lambda x: user_int_ids[x])
    test_df.product_id = test_df.product_id.map(lambda x: product_int_ids[x])
    test_df.query_id = test_df.query_id.map(lambda x: query_int_ids[x])

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
    # columns_to_dummies = [
    # 'week', 'week_day', 'device_category', 'device_platform', 'user_tier',
    # 'user_country'
    # ]
    # test_df = pd.concat(
    # [test_df, pd.get_dummies(test_df[column], prefix=column)], axis=1)
    # del test_df[column]
    # columns_to_dummies = [
    # 'season', 'collection', 'category_id_l1', 'category_id_l2',
    # 'category_id_l3', 'brand_id', 'season_year'
    # ]
    # for column in columns_to_dummies:
    # attributes_df = pd.concat([
    # attributes_df,
    # pd.get_dummies(attributes_df[column], prefix=column)
    # ],
    # axis=1)
    # del attributes_df[column]
    return train_df, test_df, attributes_df, user_int_ids, product_int_ids, query_int_ids


def create_dummies(df, columns):
    tmp_df = df.copy()
    for column in columns:
        tmp_df = pd.concat(
            [tmp_df, pd.get_dummies(tmp_df[column], prefix=column)], axis=1)
        del tmp_df[column]
    return tmp_df


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


def pickle_save(obj, file_name):
    create_path_to_file(file_name)
    pickle.dump(obj, open(file_name, 'wb'))


def pickle_load(file_name):
    return pickle.load(open(file_name, 'rb'))


def one_split_query(df, train_rate):
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


def leave_one_out(df):
    # print(df)
    df_og = df
    df = df.copy()
    df['id'] = np.arange(len(df))
    # print(df.set_index('user_id').sort_values(['timestamp'])['id'].to_dict()[317])
    user_df = df.sort_values(['timestamp'
                             ]).groupby('user_id')['id'].apply(list).to_dict()
    # print(user_df)
    users_id = df.user_id.unique()
    test_indexes = []
    for user_id in tqdm(users_id, desc='Leave one out construction'):
        # print(user_df.loc[user_id].iloc[-1])
        # print(user_df.loc[user_id].iloc[-1]['index'])
        # print(user_df.loc[user_id])
        # print(user_df[user_id])
        # if isinstance(user_df[user_id],list):
        test_indexes.append(user_df[user_id][-1])
        # else:
        # test_indexes.append(user_df[user_id])

    # test_queries_id = set(test_queries_id)
    condition = np.zeros(len(df_og), dtype=bool)
    condition[test_indexes] = 1
    # condition = df_og.index.isin(test_indexes)
    train_df = df_og.loc[~condition]
    test_df = df_og.loc[condition]
    # train_size = int(train_rate*len(df))
    # test_size = len(df)-train_size
    return train_df, test_df


def get_df_columns_with_pattern(df, pattern):
    return [c for c in df.columns if re.match(pattern, c)]


def select_top_features(df, columns, k=32):
    selectkbest = sklearn.feature_selection.SelectKBest(
        sklearn.feature_selection.chi2, k=k)
    selected_features_values = selectkbest.fit_transform(
        df[columns], df.is_click)
    selected_columns = [
        v1 for v1, v2 in zip(columns, selectkbest.get_support()) if v2 == True
    ]
    # df = df.drop(items_columns, axis=1)
    # df[selected_columns] = selected_features_values
    return selected_columns


def dimensionality_reduction(df, k=10, columns_name=""):
    decomposition = sklearn.decomposition.PCA(k)
    result = decomposition.fit_transform(df)
    result = pd.DataFrame(result)
    num_col = len(result.columns)
    result.columns = [columns_name + "_" + str(i) for i in range(0, num_col)]
    return result


def sample_fixed_size(df, num_samples):
    return df.sample(num_samples)


# class Dataset:
# def __init__(self,train_path, validation_path, attributes_path) -> None:
# self.train_path = train_path

# self.validation_path = validation_path
# self.attributes_path = attributes_path
# pass

# class SplitDataset:
# def __init__(self,train_path, validation_path, attributes_path) -> None:
# self.train_path = train_path

# self.validation_path = validation_path
# self.attributes_path = attributes_path
# pass
# def get_dataset_id(name,parameters):
# return '{%s:'%name+json.dumps(parameters, separators=(',', ':'))+'}'


def get_dataset_id(parameters):
    return json.dumps(parameters, separators=(',', ':'))


# def dataset_settings_factory(name,parameters):
def dataset_settings_factory(parameters):
    name = list(parameters.keys())[0]
    pv = list(parameters.values())[0]
    pv = parameters
    dataset_id = get_dataset_id(parameters)
    if name == 'farfetch':
        return {
            'train_path': 'data_phase1/train.parquet',
            'validation_path': 'data_phase1/validation.parquet',
            'attributes_path': 'data_phase1/attributes.parquet'
        }
    elif name == 'amazon_fashion':
        return {
            'interactions_path': 'data/AMAZON_FASHION.json.gz',
            'items_path': 'data/meta_AMAZON_FASHION.json.gz',
        }
    elif name == 'amazon_cloth':
        return {
            'interactions_path': 'data/Clothing_Shoes_and_Jewelry.json.gz',
            # 'items_path': 'data/meta_AMAZON_FASHION.json.gz',
        }
    elif name == 'farfetchfinal':
        return {
            'train_path': 'data_phase1/train.parquet',
            'validation_path': 'data_phase2/test.parquet',
            'attributes_path': 'data_phase1/attributes.parquet'
        }
    # elif name == 'farfetchdummies':
    # return {
    # 'train_path':
    # 'data_phase1/data/dummies/train.parquet',
    # 'validation_path':
    # 'data_phase1/data/dummies/validation.parquet',
    # 'attributes_path':
    # 'data_phase1/data/dummies/attributes.parquet',
    # 'user_int_ids':
    # 'data_phase1/data/dummies/user_int_ids.pickle',
    # 'product_int_ids':
    # 'data_phase1/data/dummies/product_int_ids.pickle',
    # 'query_int_ids':
    # 'data_phase1/data/dummies/query_int_ids.pickle',
    # }
    elif name == 'split':
        return {
            'train_path':
                'data_phase1/data/{}_train.parquet'.format(dataset_id),
            'validation_path':  # 'data_phase1/data/dummies/validation.parquet',
                'data_phase1/data/{}_validation.parquet'.format(dataset_id),
            'attributes_path':  # 'data_phase1/data/dummies/attributes.parquet',
                'data_phase1/attributes.parquet',
            'user_int_ids':
                None,
            'product_int_ids':
                None,
            'query_int_ids':
                None,
        }
    elif name == 'qrec':
        return {
            'train_path':
                'data_phase1/data/{}_train.csv'.format(dataset_id),
            'validation_path':  # 'data_phase1/data/dummies/validation.parquet',
                'data_phase1/data/{}_validation.csv'.format(dataset_id),
            'attributes_path':  # 'data_phase1/data/dummies/attributes.parquet',
                'data_phase1/data/{}_attributes.parquet'.format(dataset_id),
            'user_int_ids':
                'data_phase1/data/{}_user_int_ids.parquet'.format(dataset_id),
            'product_int_ids':
                'data_phase1/data/{}_product_int_ids.parquet'.format(dataset_id
                                                                    ),
            'query_int_ids':
                'data_phase1/data/{}_query_int_ids.parquet'.format(dataset_id),
        }
    elif name == 'preprocess':
        return {
            'interactions_path':
                'data/{}_preprocessed.parquet'.format(dataset_id),
        }
    elif name == 'sample':
        return {
            'interactions_path':
                'data/{}_sampled.parquet'.format(dataset_id),
        }
    else:
        return {
            'train_path':
                'data_phase1/data/{}_train.parquet'.format(dataset_id),
            'validation_path':  # 'data_phase1/data/dummies/validation.parquet',
                'data_phase1/data/{}_validation.parquet'.format(dataset_id),
            'attributes_path':  # 'data_phase1/data/dummies/attributes.parquet',
                'data_phase1/data/{}_attributes.parquet'.format(dataset_id),
            'user_int_ids':
                'data_phase1/data/{}_user_int_ids.parquet'.format(dataset_id),
            'product_int_ids':
                'data_phase1/data/{}_product_int_ids.parquet'.format(dataset_id
                                                                    ),
            'query_int_ids':
                'data_phase1/data/{}_query_int_ids.parquet'.format(dataset_id),
        }


def preprocess(dataset_input_parameters, dataset_output_parameters):

    np.random.seed(constants.RANDOM_SEED)
    dataset_input_settings = dataset_settings_factory(dataset_input_parameters)
    dataset_output_settings = dataset_settings_factory(
        dataset_output_parameters)
    interactions_df = pd.DataFrame()

    if list(dataset_input_parameters.keys())[0] == 'amazon_fashion':

        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield json.loads(l)

        def getDF(path):
            i = 0
            df = {}
            for d in parse(path):
                df[i] = d
                i += 1
            return pd.DataFrame.from_dict(df, orient='index')

        interactions_df = getDF(dataset_input_settings['interactions_path'])

        interactions_df = interactions_df.rename(
            columns={
                'overall': 'target',
                'unixReviewTime': 'timestamp',
                'reviewerID': 'user_id',
                'asin': 'item_id'
            })

        print('interactions', interactions_df.shape)
        print('users', interactions_df.user_id.nunique())
        print('items', interactions_df.item_id.nunique())
        print(
            interactions_df.groupby('user_id')
            ['item_id'].count().value_counts()[:10])
        interactions_df = interactions_df.loc[interactions_df.target >= 4]
        interactions_df.target = 1
        # interactions_df.loc[interactions_df.target<4] = 0
        # interactions_df.loc[interactions_df.target>=4] = 1

        datetime = pd.to_datetime(interactions_df.reviewTime)
        days = datetime.dt.day_name().astype('category').cat.codes
        days.name = 'day'
        interactions_df = pd.concat([interactions_df, days], axis=1)
        weeks = datetime.dt.isocalendar().week
        interactions_df = pd.concat([interactions_df, weeks], axis=1)
        interactions_df.timestamp = interactions_df.timestamp - interactions_df.timestamp.min(
        )

        users_history_size = interactions_df.groupby(
            'user_id')['item_id'].count()
        users_history_size = users_history_size.loc[users_history_size >= list(dataset_output_parameters.values())[0]['mshi']]
        interactions_df = interactions_df.loc[interactions_df.user_id.isin(
            users_history_size.index)]

        interactions_df.item_id = interactions_df.item_id.astype(
            'category').cat.codes
        interactions_df.user_id = interactions_df.user_id.astype(
            'category').cat.codes
    elif list(dataset_input_parameters.keys())[0] == 'amazon_cloth':

        interactions_df = pd.read_parquet('data/amazon_clothing.parquet')
        # users_history_size = interactions_df.groupby(
            # 'reviewerID')['asin'].count()
        # users_history_size = users_history_size.loc[users_history_size >= 5]
        # interactions_df = interactions_df.loc[interactions_df.reviewerID.isin(
            # users_history_size.index)]

        filtered_counts = interactions_df[['reviewerID', 'asin'
                                          ]].groupby('reviewerID').count()
        counts_probabilities = filtered_counts.value_counts(
        ) / filtered_counts.value_counts().sum()
        num_users = 10000

        num_interactions_users = filtered_counts.reset_index().groupby(
            'asin').agg(list)['reviewerID'].to_dict()
        final_users = []
        samples = []
        while len(final_users) < num_users:
        # for sample in samples:
            if len(samples) == 0:
                samples = list(np.random.choice(
                    counts_probabilities.reset_index()['asin'].to_numpy(),
                    num_users,
                    p=counts_probabilities.values))
            user = num_interactions_users[samples.pop(0)].pop(0)

            final_users.append(user)
        print('final users',len(final_users))

        interactions_df = interactions_df[interactions_df['reviewerID'].isin(
            set(final_users))]

        # interactions_df=getDF(dataset_input_settings['interactions_path'])

        interactions_df = interactions_df.rename(
            columns={
                'overall': 'target',
                'unixReviewTime': 'timestamp',
                'reviewerID': 'user_id',
                'asin': 'item_id'
            })
        interactions_df = interactions_df.loc[interactions_df.target >= 4]
        interactions_df.target = 1
        # interactions_df.loc[interactions_df.target<4] = 0
        # interactions_df.loc[interactions_df.target>=4] = 1

        datetime = pd.to_datetime(interactions_df.reviewTime)
        days = datetime.dt.day_name().astype('category').cat.codes
        days.name = 'day'
        interactions_df = pd.concat([interactions_df, days], axis=1)
        weeks = datetime.dt.isocalendar().week
        interactions_df = pd.concat([interactions_df, weeks], axis=1)
        interactions_df.timestamp = interactions_df.timestamp - interactions_df.timestamp.min(
        )

        users_history_size = interactions_df.groupby(
            'user_id')['item_id'].count()
        users_history_size = users_history_size.loc[users_history_size >= list(dataset_output_parameters.values())[0]['mshi']]
        interactions_df = interactions_df.loc[interactions_df.user_id.isin(
            users_history_size.index)]

        interactions_df.item_id = interactions_df.item_id.astype(
            'category').cat.codes
        interactions_df.user_id = interactions_df.user_id.astype(
            'category').cat.codes
    elif list(dataset_output_parameters.keys())[0] == 'sample':
        # print(dataset_input_settings['interactions_path'])
        interactions_df = pd.read_parquet(dataset_input_settings['interactions_path'])
        # interactions_df = interactions_df.sample(len(interactions_df)*list(dataset_output_parameters.values())[0]['rate'])
        # print(interactions_df)
        interactions_df, _ = leave_one_out(interactions_df)
        # print(interactions_df)

    # print(list(dataset_input_parameters.keys())[0])
    # interactions_df.to_parquet(dataset_output_settings['interactions_path'])
    return interactions_df


def train_test_split(interactions_df, split_parameters):
    if list(split_parameters.keys())[0] == 'one_split':
        train_df, test_df = one_split(interactions_df,
                                      split_parameters['train_rate'])

    return train_df, test_df
