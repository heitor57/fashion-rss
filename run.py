import pandas as pd
import re
import joblib
from constants import dataset_parameters
import torch
import value_functions
import os.path
import constants
import neural_networks
import loss_functions
import recommenders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import dataset
import utils
import argparse

dataset_parameters = {
    # 'rate': constants.RATE,
    # 'random_seed': constants.RANDOM_SEED,
    # 'train_path_name': 'data_phase1/train.parquet',
    # 'test_path_name': 'data_phase1/validation.parquet',
    # 'attributes_path_name': 'data_phase1/attributes.parquet',
    'train_path_name': 'data_phase1/data/dummies/train.parquet',
    'test_path_name': 'data_phase1/data/dummies/test.parquet',
    'attributes_path_name': 'data_phase1/data/dummies/attributes.parquet',
    # 'train_path_name': 'data_phase1/data/train.parquet',
    # 'test_path_name': 'data_phase1/data/test.parquet',
    'user_int_ids': 'data_phase1/data/dummies/user_int_ids.pickle',
    'product_int_ids': 'data_phase1/data/dummies/product_int_ids.pickle',
}
argparser = argparse.ArgumentParser()
argparser.add_argument('-m', type=str)
args = argparser.parse_args()
method = args.m

dataset_parameters_id = joblib.hash(dataset_parameters)
# negative_file_name = 'negative_samples_'+dataset_parameters_id
# parameters_id = utils.parameters_to_str(parameters)

# train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
# dataset.parquet_load(file_name=dataset_parameters['train_path_name']),
# dataset.parquet_load(file_name=dataset_parameters['test_path_name']),
# dataset.parquet_load(file_name=dataset_parameters['attributes_path_name']))
train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = (
    dataset.parquet_load(file_name=dataset_parameters['train_path_name']),
    dataset.parquet_load(file_name=dataset_parameters['test_path_name']),
    dataset.parquet_load(file_name=dataset_parameters['attributes_path_name']),
    dataset.pickle_load(file_name=dataset_parameters['user_int_ids']),
    dataset.pickle_load(file_name=dataset_parameters['product_int_ids']))
# print(train_normalized_df)

num_users = len(user_int_ids)
num_items = len(product_int_ids)
if method == 'ewqeq':

    loss_function = loss_functions.BPRLoss(1e-4, 0.001)
    nn = neural_networks.BilinearNet(num_users,
                                     num_items,
                                     constants.EMBEDDING_DIM,
                                     sparse=False)
    # nn = neural_networks.PoolNet(num_items,constants.EMBEDDING_DIM)
    # nn = neural_networks.PopularityNet(num_items)
    nnvf = value_functions.NNVF(nn, loss_function)

    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'Random':
    vf = value_functions.RandomVF()
    recommender = recommenders.SimpleRecommender(vf, name=method)
elif method == 'PopularityNet':
    loss_function = loss_functions.BPRLoss(1e-4, 0.001)
    nn = neural_networks.PopularityNet(num_items)
    nnvf = value_functions.NNVF(nn,
                                loss_function,
                                num_batchs=2000,
                                batch_size=2048)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'ContextualPopularityNet':
    loss_function = loss_functions.BPRLoss(1e-4, 0.001)
    items_columns = list(map(str,list(range(0,32))))
    # items_columns = [
        # 'season', 'collection', 'category_id_l1', 'category_id_l2',
        # 'category_id_l3', 'brand_id', 'season_year'
    # ]
    pattern = '|'.join(items_columns)
    items_columns = [c for c in attributes_df.columns if re.match(pattern, c)]
    users_columns = [
        'week', 'week_day', 'device_category', 'device_platform', 'user_tier',
        'user_country'
    ]
    pattern = '|'.join(users_columns)
    users_columns = [
        c for c in train_normalized_df.columns if re.match(pattern, c)
    ]

    nn = neural_networks.ContextualPopularityNet(num_items,
                                                 attributes_df[items_columns],
                                                 users_columns)
    nnvf = value_functions.NNVF(nn,
                                loss_function,
                                num_batchs=20000,
                                batch_size=2048)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)

results = []
product_str_ids = {v: k for k, v in product_int_ids.items()}
for name, group in tqdm(test_normalized_df.groupby('query_id')):

    if method == 'ContextualPopularityNet':
        users, items = recommender.recommend(group['user_id'].to_numpy(),
                                             group['product_id'].to_numpy(), users_context=group[users_columns])
    else:
        users, items = recommender.recommend(group['user_id'].to_numpy(),
                                             group['product_id'].to_numpy())
    user_id = group['user_id'].iloc[0]
    group = group.set_index('user_id')
    query_id = group['query_id'].iloc[0]
    j = 1
    # print(items)
    for i in items:
        # print(i)
        results.append([query_id, product_str_ids[i], j])
        j += 1
    # recommender.recommend()

results_df = pd.DataFrame(results, columns=['query_id', 'product_id', 'rank'])
results_df.to_csv(f'data_phase1/data/{method}_output.csv', index=False)
