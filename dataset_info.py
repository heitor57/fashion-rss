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

# train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
# dataset.parquet_load(file_name=f'data_phase1/train.parquet'),
# # dataset.parquet_load(file_name=f'data_phase1/data/train_one_split.parquet'),
# dataset.parquet_load(file_name='data_phase1/validation.parquet'),
# # dataset.parquet_load(file_name='data_phase1/data/test_one_split.parquet'),
# dataset.parquet_load(file_name='data_phase1/attributes.parquet'))

# train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
train_df, validation_df, attributes_df = dataset.parquet_load(
    file_name=f'data_phase1/data/dummies/train.parquet'), dataset.parquet_load(
        file_name='data_phase1/data/dummies/validation.parquet'
    ), dataset.parquet_load(
        file_name='data_phase1/data/dummies/attributes.parquet')
# dataset.parquet_load(file_name='data_phase1/data/test_one_split.parquet'),
# dataset.parquet_load(file_name=f'data_phase1/data/train_one_split.parquet'),

num_users = len(set(train_df.user_id.unique()) | set(validation_df.user_id.unique()))
num_items = len(attributes_df)
total_observations = len(train_df)+len(validation_df)


print('num cold start users', len( set(validation_df.user_id.unique())-set(train_df.user_id.unique()))/len(set(validation_df.user_id.unique())))
print('num cold start items', len( set(validation_df.product_id.unique())-set(train_df.product_id.unique()))/len(set(validation_df.product_id.unique())))

print('Num users: {} Num items: {} Num user-item pair unique observations: {}'.
      format(num_users, num_items, total_observations))
print('Train Sparsity: {:.7%}'.format(1 - total_observations /
                                      (num_users * num_items)))
print('Users unique items clicked {}'.format(
    len(train_df.product_id.unique())))

train_df = dataset.parquet_load(file_name=f'data_phase1/train.parquet')
print('number of sessions per user count')
print(train_df.groupby('user_id')['session_id'].nunique().value_counts())
print('number of queries per user count')
print(train_df.groupby('user_id')['query_id'].nunique().value_counts())
# print('Users unique items consumed {}'.format())
print(train_df.describe())

print(train_df[[
    'page_type', 'previous_page_type', 'device_category', 'device_platform',
    'user_tier', 'user_country', 'context_type', 'context_value'
]].nunique())

print(train_df['context_type'].unique())


