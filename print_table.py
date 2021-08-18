from collections import defaultdict
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


def statistic_test(x, y, p):
    statistic, pvalue = scipy.stats.ttest_ind(x, y)
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    if pvalue < p:
        if x_mean > y_mean:
            return 'gain'
        else:
            return 'loss'
    else:
        return 'tie'


METHODS_PRETTY_NAME = {
    'bi': 'BilinearNet',
    'ncf': 'NeuMF',
    'svd': 'SVD',
    'svdpp': 'SVD++',
    'popular': 'Popular',
    'random': 'Random',
    'lightgcn': 'LightGCN',
    'coverage': 'Coverage',
    'stacking': 'MLP(LightGCN+Popular)',
}

stat_result_symbols = {'gain': "↑", 'loss': "↓", 'tie': "⏺"}

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', nargs='*')
argparser.add_argument('-r', default='stacking')
args = argparser.parse_args()

# dataset_input_parameters = {'amazon_fashion': {}}
# dataset_input_parameters = {'amazon_cloth': {}}

# dataset_input_parameters = {'preprocess': {'base': dataset_input_parameters}}
dataset_name = 'amazon_fashion'
# dataset_name = 'amazon_cloth'
mshi = 5
# mshi = 10
dataset_input_parameters = {dataset_name: {}}

dataset_input_parameters = {
    'preprocess': {
        'base': dataset_input_parameters,
        'mshi': mshi
    }
}

dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

methods_metrics_values = {}
best_parameters = utils.load_best_parameters()
for method in args.m:

    execution_id = joblib.hash(
        (method, best_parameters[dataset_name][mshi][method], dataset_input_parameters,5))
    # path = f'data/metrics/mrr/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
    path = f'data/metrics/mrr/{execution_id}_output.csv'
    mrrs = pd.read_csv(path)['0']

    # path = f'data/metrics/ndcg/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
    path = f'data/metrics/ndcg/{execution_id}_output.csv'
    ndcgs = pd.read_csv(path)['0']

    path = f'data/metrics/hit/{execution_id}_output.csv'
    hits = pd.read_csv(path)['0']
    print(hits)
    methods_metrics_values[method] = {'MRR': mrrs, 'NDCG': ndcgs, 'Hits': hits}

metrics_names = list(list(methods_metrics_values.values())[0].keys())
num_metrics = len(metrics_names)
num_methods = len(args.m)

metrics_final_results = {}
for metric_name in metrics_names:
    tmp_ref_metrics = methods_metrics_values.pop(args.r)
    methods_metric_mean = {
        method: np.mean(metrics_values[metric_name])
        for method, metrics_values in methods_metrics_values.items()
    }
    top_method = max(methods_metric_mean.items(), key=operator.itemgetter(1))[0]
    methods_metrics_values[args.r] = tmp_ref_metrics

    stat_result = statistic_test(
        methods_metrics_values[args.r][metric_name],
        methods_metrics_values[top_method][metric_name], 0.05)
    stat_result = stat_result_symbols[stat_result]

    metrics_final_results[metric_name] = {
        method: '{:.5f}{}'.format(np.mean(metrics_values[metric_name]),
                                  (stat_result if method == args.r else ''))
        for method, metrics_values in methods_metrics_values.items()
    }

# final_table = [[''] * (num_metrics + 1)] * (num_methods + 1)
final_table = [
    ['' for __ in range(num_metrics + 1)] for _ in range(num_methods + 1)
]

# i = 0
# for method in args.m:
# final_table[1 + i][0] = method
# i += 1

# for method in args.m:
# final_table[1 + i][0] = method
# i += 1
i = 0
for metric_name, methods_values in metrics_final_results.items():
    j = 0
    final_table[0][1 + i] = metric_name
    for method in args.m:
        final_table[1 + j][0] = METHODS_PRETTY_NAME[method]
        final_table[1 + j][1 + i] = methods_values[method]
        j += 1
    i += 1

print(tabulate(final_table))
