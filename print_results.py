from collections import defaultdict
import yaml
import parameters
import scipy.stats
import operator
import scipy.sparse
import pandas as pd
import sklearn.ensemble
import sklearn.tree
import sklearn.model_selection
import sklearn.neural_network
from tabulate import tabulate
try:
    import spotlight
    from spotlight.sequence.implicit import ImplicitSequenceModel
except ModuleNotFoundError:
    pass

import seaborn
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
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', nargs='*')
args = argparser.parse_args()
best_parameters = utils.load_best_parameters()

# for dataset_name in ['amazon_fashion','amazon_cloth']:
for dataset_name in ['amazon_fashion']:
    for mshi in [5,10]:
        dataset_input_parameters = {dataset_name: {}}

        dataset_input_parameters = {
            'preprocess': {
                'base': dataset_input_parameters,
                'mshi': mshi
            }
        }
        # print(dataset_name,mshi)

        dataset_input_settings = dataset.dataset_settings_factory(
            dataset_input_parameters)

        interactions_df = dataset.parquet_load(
            dataset_input_settings['interactions_path'])
        num_users = interactions_df.user_id.max() + 1
        num_items = interactions_df.item_id.max() + 1
        train_df, test_df = dataset.leave_one_out(interactions_df)
        methods_metrics_values = defaultdict(dict)
        num_executions = 1
        for method in args.m:
            method_search_parameters = []
            create_method = lambda x: None
            if method == 'svd':
                method_search_parameters = parameters.SVD_PARAMETERS
                create_method = parameters.create_svd
            elif method == 'svdpp':
                method_search_parameters = parameters.SVDPP_PARAMETERS
                create_method = parameters.create_svdpp
            elif method == 'ncf':
                method_search_parameters = parameters.NCF_PARAMETERS
                create_method = parameters.create_ncf
            elif method == 'bi':
                method_search_parameters = parameters.BI_PARAMETERS
                create_method = parameters.create_bi
            elif method == 'lightgcn':
                method_search_parameters = parameters.LIGHTGCN_PARAMETERS
                create_method = parameters.create_lightgcn
            else:
                raise NameError
            # method_search_parameters=[utils.dict_union(msp, {'num_users':num_users,'num_items':num_items,'scootensor':scootensor,'num_batchs':200,'batch_size': len(train_df)//2}) for msp in method_search_parameters]

            for method_parameters in method_search_parameters:
                # print(method_parameters)
                # print((method, method_parameters,
                                            # dataset_input_parameters, num_executions))
                execution_id = joblib.hash((method, method_parameters,
                                            dataset_input_parameters, num_executions))
                # print(execution_id)
                # raise SystemExit

                path = f'data/metrics/mrr/{execution_id}_output.csv'
                mrrs = pd.read_csv(path)['0']

                path = f'data/metrics/ndcg/{execution_id}_output.csv'
                ndcgs = pd.read_csv(path)['0']

                path = f'data/metrics/hit/{execution_id}_output.csv'
                hits = pd.read_csv(path)['0']
                # print(hits)
                methods_metrics_values[method][str(method_parameters)] = {
                    'MRR': mrrs,
                    'NDCG': ndcgs,
                    'Hits': hits
                }

        metrics_names = list(
            list(list(methods_metrics_values.values())[0].values())[0].keys())
        num_metrics = len(metrics_names)
        num_methods = len(args.m)
        metric = 'MRR'
        for method in args.m:
            d= {k: np.mean(v[metric]) for k, v in methods_metrics_values[method].items()}
            d={k: v for k, v in sorted(d.items(), key=lambda item: item[1],reverse=True)}
            # print(list(d.keys())[0],list(d.values())[0])
            best_parameters[dataset_name][mshi][method] = eval(list(d.keys())[0])
            # for k, v in d.items():
                # print(k,v)

utils.save_best_parameters(best_parameters)
# print(yaml.dump(best_parameters))
