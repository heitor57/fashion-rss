import pandas as pd
import joblib
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

dataset_input_parameters = {
    'dummies': {
        'base': {
            'split': {
                'base': {
                    'farfetch': {}
                },
                'train_size': 0.9
            }
        }
    }
}
dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

train_df, test_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = (
    dataset.parquet_load(file_name=dataset_input_settings['train_path']),
    dataset.parquet_load(file_name=dataset_input_settings['validation_path']),
    dataset.parquet_load(file_name=dataset_input_settings['attributes_path']),
    dataset.pickle_load(file_name=dataset_input_settings['user_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['product_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['query_int_ids']),
)

num_users = len(set(train_df.user_id.unique()) | set(test_df.user_id.unique()))
num_items = len(attributes_df)
total_observations = len(train_df)+len(test_df)


print('num cold start users', len( set(test_df.user_id.unique())-set(train_df.user_id.unique()))/len(set(test_df.user_id.unique())))
print('num cold start items', len( set(test_df.product_id.unique())-set(train_df.product_id.unique()))/len(set(test_df.product_id.unique())))

print('Num users: {} Num items: {} Num user-item pair unique observations: {}'.
      format(num_users, num_items, total_observations))
print('Train Sparsity: {:.7%}'.format(1 - total_observations /
                                      (num_users * num_items)))
print('Users unique items clicked {}'.format(
    len(train_df.product_id.unique())))

# train_df = dataset.parquet_load(file_name=f'data_phase1/train.parquet')
# print('number of sessions per user count')
# print(train_df.groupby('user_id')['session_id'].nunique().value_counts())
# print('number of queries per user count')
# print(train_df.groupby('user_id')['query_id'].nunique().value_counts())
# print('Users unique items consumed {}'.format())
# print(train_df.describe())

# print(train_df[[
    # 'page_type', 'previous_page_type', 'device_category', 'device_platform',
    # 'user_tier', 'user_country', 'context_type', 'context_value'
# ]].nunique())

# print(train_df['context_type'].unique())


