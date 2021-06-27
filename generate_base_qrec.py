import pandas as pd
import re
import joblib
from torch.optim import optimizer
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

dataset_1_parameters = {'farfetch': {}}
# dataset_output_name= 'split'
dataset_input_parameters = {
    'split': {
        'base': dataset_1_parameters,
        'train_size': 0.8
    }
}
dataset_input_parameters = {'dummies': {'base': dataset_input_parameters}}
dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

dataset_output_parameters = {'qrec':{'base':dataset_input_parameters}}
dataset_output_settings = dataset.dataset_settings_factory(
    dataset_output_parameters)

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = (
    dataset.parquet_load(file_name=dataset_input_settings['train_path']),
    dataset.parquet_load(file_name=dataset_input_settings['validation_path']),
    dataset.parquet_load(file_name=dataset_input_settings['attributes_path']),
    dataset.pickle_load(file_name=dataset_input_settings['user_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['product_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['query_int_ids']),
)

print(train_normalized_df,test_normalized_df)

tmp_train_df = train_normalized_df.copy()
tmp_train_df = tmp_train_df.loc[tmp_train_df.is_click > 0]
tmp_train_df = tmp_train_df.groupby(['user_id', 'product_id'
                                    ])['is_click'].sum() >= 1
tmp_train_df = tmp_train_df.reset_index()
tmp_train_df.is_click = tmp_train_df.is_click.astype(int)

# train_normalized_df[['user_id','product_id','is_click']]
tmp_train_df.to_csv(dataset_output_settings['train_path'],header=None,index=None)

tmp_train_df = test_normalized_df.copy()
tmp_train_df = tmp_train_df.loc[tmp_train_df.is_click > 0]
tmp_train_df = tmp_train_df.groupby(['user_id', 'product_id'
                                    ])['is_click'].sum() >= 1
tmp_train_df = tmp_train_df.reset_index()
tmp_train_df.is_click = tmp_train_df.is_click.astype(int)
tmp_train_df.to_csv(dataset_output_settings['validation_path'],header=None,index=None)
