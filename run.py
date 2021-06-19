import pandas as pd
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

parameters = {'rate': constants.RATE, 'random_seed': constants.RANDOM_SEED}
parameters_id = utils.parameters_to_str(parameters)

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
    dataset.parquet_load(file_name=f'data_phase1/train.parquet'),
    # dataset.parquet_load(file_name=f'data_phase1/data/train_one_split.parquet'),
    dataset.parquet_load(file_name='data_phase1/validation.parquet'),
    # dataset.parquet_load(file_name='data_phase1/data/test_one_split.parquet'),
    dataset.parquet_load(file_name='data_phase1/attributes.parquet'))

if os.path.isfile(constants.negative_samples_file):
    train_normalized_df = dataset.parquet_load(constants.negative_samples_file)
else:
    train_normalized_df = dataset.negative_samples(train_normalized_df)
    dataset.parquet_save(train_normalized_df, constants.negative_samples_file)
num_users = len(user_int_ids)
num_items = len(product_int_ids)
loss_function = loss_functions.BPRLoss(1e-4, 0.001)
# nn = neural_networks.BilinearNet(num_users,
                                 # num_items,
                                 # constants.EMBEDDING_DIM,
                                 # sparse=False)
nn = neural_networks.PoolNet(num_items,constants.EMBEDDING_DIM)
nnvf = value_functions.NNVF(nn, loss_function)

recommender = recommenders.NNRecommender(nnvf, name="NN")
recommender.train(train_normalized_df)

results = []
product_str_ids = {v: k for k, v in product_int_ids.items()}
for name, group in tqdm(test_normalized_df.groupby('query_id')):
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
results_df.to_csv('data_phase1/data/output.csv', index=False)
