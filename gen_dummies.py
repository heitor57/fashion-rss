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

dataset_input_parameters = {'farfetchfinal':{}}
# dataset_input_parameters = {'split':{'base': {'farfetch':{}},'train_size':0.8 }}
dataset_input_settings = dataset.dataset_settings_factory(dataset_input_parameters)

dataset_output_parameters = {'dummies':{'base': dataset_input_parameters}}
dataset_output_settings = dataset.dataset_settings_factory(dataset_output_parameters)


train_df, test_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = dataset.farfetch_train_test_normalization(
	# dataset.parquet_load(file_name=f'data_phase1/train.parquet'),
	dataset.parquet_load(file_name=dataset_input_settings['train_path']),
	# dataset.parquet_load(file_name='data_phase1/validation.parquet'),
	dataset.parquet_load(file_name=dataset_input_settings['validation_path']),
	dataset.parquet_load(file_name=dataset_input_settings['attributes_path']))

users_columns_to_dummies = [
	# 'week', 'week_day',
	'device_category', 'device_platform',
	'user_tier',
	'user_country'
]

test_df['is_test'] = 1
train_df['is_test'] = 0

train_test_df = dataset.create_dummies(pd.concat([train_df,test_df],axis=0),users_columns_to_dummies)
test_df = train_test_df.loc[train_test_df['is_test'] == 1].copy()
train_df = train_test_df.loc[train_test_df['is_test'] == 0].copy()
del test_df['is_test'], train_df['is_test']

# items_columns_to_dummies = [
# 	'season', 'collection','gender','category_id_l1','category_id_l2', 'attribute_values'
# ]
# pattern = '|'.join(users_columns_to_dummies)
# columns = dataset.get_df_columns_with_pattern(train_df,
													# pattern)

items_columns_to_dummies = [
    'season', 'collection','gender','category_id_l1','category_id_l2'
]

attributes_df = dataset.create_dummies(attributes_df, items_columns_to_dummies)

# items_columns = []
# items_columns_to_dummies = ['attribute_values']
# attributes_df = dataset.create_dummies(attributes_df, items_columns_to_dummies)
# pattern = '|'.join(items_columns_to_dummies)
# items_columns.append(dataset.get_df_columns_with_pattern(attributes_df, pattern))
# attributes_df = pd.concat([attributes_df.drop(items_columns[0],axis=1),dataset.dimensionality_reduction(attributes_df[items_columns[0]], columns_name="attribute_values").astype(np.float32)],axis=1)

# items_columns_to_dummies = ['material_values']
# attributes_df = dataset.create_dummies(attributes_df, items_columns_to_dummies)
# pattern = '|'.join(items_columns_to_dummies)
# items_columns.append(dataset.get_df_columns_with_pattern(attributes_df, pattern))
# attributes_df = pd.concat([attributes_df.drop(items_columns[1],axis=1),dataset.dimensionality_reduction(attributes_df[items_columns[1]], columns_name="material_values").astype(np.float32)],axis=1)

# print(attributes_df.columns)
# input("	")
# for
# for column in columns[]
# train_df[]

# selected_columns = dataset.select_top_features(train_normalized_df,
												  # columns)
# train_normalized_df = pd.concat([train_normalized_df.drop(columns, axis=1),train_normalized_df[selected_columns]],axis=
dataset.parquet_save(train_df,
					 dataset_output_settings['train_path'])

# test_normalized_df = dataset.select_top_features(test_normalized_df,
												 # users_columns_to_dummies)

# test_normalized_df = pd.concat([test_normalized_df.drop(columns, axis=1),test_normalized_df[selected_columns]],axis=1)
dataset.parquet_save(test_df,
					 dataset_output_settings['validation_path'])

# attributes_df = dataset.select_top_features(attributes_df,
# items_columns_to_dummies)

# attributes_df = pd.concat([attributes_df.drop(items_columns,axis=1),dataset.dimensionality_reduction(attributes_df[items_columns]).astype(np.float32)],axis=1)
print(attributes_df)
attributes_df.columns= list(map(str,attributes_df.columns))
dataset.parquet_save(attributes_df,
					 dataset_output_settings['attributes_path'])

dataset.pickle_save(user_int_ids, dataset_output_settings['user_int_ids'])
dataset.pickle_save(product_int_ids, dataset_output_settings['product_int_ids'])
dataset.pickle_save(query_int_ids, dataset_output_settings['query_int_ids'])
