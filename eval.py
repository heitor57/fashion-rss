
import metrics
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

dataset_1_parameters= {'farfetch':{}}
# dataset_output_name= 'split'
dataset_input_parameters = {'split':{'base': dataset_1_parameters,'train_size':0.8 }}
dataset_input_parameters = {'dummies':{'base': dataset_input_parameters}}
dataset_input_settings = dataset.dataset_settings_factory(dataset_input_parameters)

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', type=str)
args = argparser.parse_args()
method = args.m

train_df, validation_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = (
    dataset.parquet_load(file_name=dataset_input_settings['train_path']),
    dataset.parquet_load(file_name=dataset_input_settings['validation_path']),
    dataset.parquet_load(file_name=dataset_input_settings['attributes_path']),
    dataset.pickle_load(file_name=dataset_input_settings['user_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['product_int_ids']),
    dataset.pickle_load(file_name=dataset_input_settings['query_int_ids']),
    )

# print(test_normalized_df.groupby('user_id').count().mean())
# raise SystemError

# print(train_normalized_df)

num_users = len(user_int_ids)
num_items = len(product_int_ids)

# results_df = pd.DataFrame(results, columns=['query_id', 'product_id', 'rank'])
results_df =  pd.read_csv(f'data_phase1/data/{dataset.get_dataset_id(dataset_input_parameters)}_{method}_output.csv')
results_df = results_df.sort_values(['query_id','rank'])
# validation_df = validation_df.sort_values(['query_id','rank'])
print(results_df)

def _f(x,d):
    if x in d:
        return d[x]
    else:
        return x
results_df['query_id']=results_df['query_id'].map(lambda x: _f(x,query_int_ids))
# results_df['user_id']=results_df['user_id'].map(lambda x: _f(x,user_int_ids))
results_df['product_id']=results_df['product_id'].map(lambda x: _f(x,product_int_ids))

all_product_is_click = validation_df.set_index(['query_id','product_id'])['is_click'].to_dict()

# print(list(all_product_is_click.keys())[0])
# results_df
ranks = []
is_clicks = []
mrr = 0
c =0 
for name, group in tqdm(results_df.groupby('query_id')):
    
    query_id = group['query_id'].iloc[0]
    # print(all_product_is_click)
    # is_click=all_product_is_click[query_id]
    # print(is_click)
    # print(group['rank'])
    # print(group.product_id)
    # print(is_click)
    mrr+= metrics.cython_rr(group.query_id.values,group.product_id.values,group['rank'].values,all_product_is_click)
    c+=1

print(mrr)
print(c)
print('mrr',mrr/c)

# mrr = metrics.mrr(
# np.array(ranks),
# np.array(is_clicks),
# )

    # rr = metrics.rr(group['rank'].values,product_is_click)
    # rr = None
    # for prod, rank in zip(group['product_id'].values,):
        # if product_is_click.loc[prod]:
            # rr = 1/rank
            # break
    # if rr== None:
        # raise ValueError
    # rrs.append(rr)
    
# print(f'mrr: {np.mean(rrs)}')
