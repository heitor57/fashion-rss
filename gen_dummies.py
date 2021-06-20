import pandas as pd
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
from constants import dataset_parameters

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
    # dataset.parquet_load(file_name=f'data_phase1/train.parquet'),
    dataset.parquet_load(file_name=dataset_parameters['train_path_name']),
    # dataset.parquet_load(file_name='data_phase1/validation.parquet'),
    dataset.parquet_load(file_name=dataset_parameters['test_path_name']),
    dataset.parquet_load(file_name=dataset_parameters['attributes_path_name']))

users_columns_to_dummies = [
    'week', 'week_day', 'device_category', 'device_platform', 'user_tier',
    'user_country'
]
# cfname = joblib.hash(users_columns_to_dummies,
                     # dataset_parameters['train_path_name'])
# if utils.file_exists(cfname):
    # train_normalized_df = dataset.parquet_load(cfname)
# else:
train_normalized_df = dataset.create_dummies(train_normalized_df,
                                             users_columns_to_dummies)
# utils.create_path_to_file()
dataset.parquet_save(train_normalized_df,'data_phase1/data/dummies/train.parquet')
    # utils.create_path_to_file()

test_normalized_df = dataset.create_dummies(test_normalized_df,users_columns_to_dummies)
dataset.parquet_save(test_normalized_df,'data_phase1/data/dummies/test.parquet')
items_columns_to_dummies = [
'season', 'collection', 'category_id_l1', 'category_id_l2',
'category_id_l3', 'brand_id', 'season_year'
]
attributes_df = dataset.create_dummies(attributes_df,items_columns_to_dummies)
pattern = '|'.join(items_columns_to_dummies)
items_columns = [
    c for c in train_normalized_df.columns if re.match(pattern, c)
]
attributes_df[items_columns]


dataset.parquet_save(attributes_df,'data_phase1/data/dummies/attributes.parquet')

dataset.pickle_save(user_int_ids,dataset_parameters['user_int_ids'])
dataset.pickle_save(product_int_ids,dataset_parameters['product_int_ids'])
