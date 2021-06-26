import pandas as pd
import sklearn
import sklearn.feature_selection
import re
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
import argparse
from constants import settings
settings = {
    'train_path_name': 'data_phase1/data/train.parquet',
    'validation_path_name': 'data_phase1/data/validation.parquet',
    'attributes_path_name': 'data_phase1/attributes.parquet',
}

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = dataset.farfetch_train_test_normalization(
    # dataset.parquet_load(file_name=f'data_phase1/train.parquet'),
    dataset.parquet_load(file_name=settings['train_path_name']),
    # dataset.parquet_load(file_name='data_phase1/validation.parquet'),
    dataset.parquet_load(file_name=settings['validation_path_name']),
    dataset.parquet_load(file_name=settings['attributes_path_name']))

users_columns_to_dummies = [
    'week', 'week_day',
    'device_category', 'device_platform',
    'user_tier',
    # 'user_country'
]
test_normalized_df['is_test'] = 1
train_normalized_df['is_test'] = 0
train_test_df = dataset.create_dummies(pd.concat([train_normalized_df,test_normalized_df],axis=0),users_columns_to_dummies)
test_normalized_df = train_test_df.loc[train_test_df['is_test'] == 0].copy()
train_normalized_df = train_test_df.loc[train_test_df['is_test'] != 1].copy()

del test_normalized_df['is_test'], train_normalized_df['is_test']
# del test_normalized_df['is_click']

# train_normalized_df = dataset.create_dummies(train_normalized_df,
                                             # users_columns_to_dummies)
pattern = '|'.join(users_columns_to_dummies)
columns = dataset.get_df_columns_with_pattern(train_normalized_df,
                                                    pattern)
# selected_columns = dataset.select_top_features(train_normalized_df,
                                                  # columns)
# train_normalized_df = pd.concat([train_normalized_df.drop(columns, axis=1),train_normalized_df[selected_columns]],axis=
dataset.parquet_save(train_normalized_df,
                     'data_phase1/data/dummies/train.parquet')

# test_normalized_df = dataset.select_top_features(test_normalized_df,
                                                 # users_columns_to_dummies)

# test_normalized_df = pd.concat([test_normalized_df.drop(columns, axis=1),test_normalized_df[selected_columns]],axis=1)
dataset.parquet_save(test_normalized_df,
                     'data_phase1/data/dummies/validation.parquet')

items_columns_to_dummies = [
    'season', 'collection','gender','category_id_l1', 'season_year'
]
attributes_df = dataset.create_dummies(attributes_df, items_columns_to_dummies)
# attributes_df = dataset.select_top_features(attributes_df,
# items_columns_to_dummies)
pattern = '|'.join(items_columns_to_dummies)
items_columns = dataset.get_df_columns_with_pattern(attributes_df, pattern)

# attributes_df = pd.concat([attributes_df.drop(items_columns,axis=1),dataset.dimensionality_reduction(attributes_df[items_columns]).astype(np.float32)],axis=1)
print(attributes_df)
attributes_df.columns= list(map(str,attributes_df.columns))
dataset.parquet_save(attributes_df,
                     'data_phase1/data/dummies/attributes.parquet')

dataset.pickle_save(user_int_ids, 'data_phase1/data/dummies/user_int_ids.pickle')
dataset.pickle_save(product_int_ids, 'data_phase1/data/dummies/product_int_ids.pickle')
dataset.pickle_save(query_int_ids, 'data_phase1/data/dummies/query_int_ids.pickle')
