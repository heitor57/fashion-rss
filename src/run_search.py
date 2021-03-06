import parameters
from copy import copy
from collections import defaultdict
import scipy.sparse
import pandas as pd
import sklearn.ensemble
import sklearn.tree
import sklearn.model_selection
import sklearn.neural_network
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


def leave_one_out_experiment(recommender, method, dataset_name, train_df,
                             test_df, num_users, num_items, num_negatives,
                             users, interactions_matrix, num_executions,
                             execution_id, interactions_df):
    # recommender = create_method(method_parameters)
    train_method(
        recommender, method, {
            'name': dataset_name,
            'train': train_df,
            'num_users': num_users,
            'num_items': num_items
        })

    mrrs = []
    ndcgs = []
    hits = []

    for i in range(num_executions):
        seed = i
        np.random.seed(seed)
        negatives_id = joblib.hash((seed, test_df, users, num_negatives,
                                    interactions_matrix, num_items))
        fpath = f'data/utils/negatives/{negatives_id}'
        path = f'data/results/{execution_id}_{i}_output.csv'
        if not utils.file_exists(fpath):
            negatives = utils.generate_negative_samples(test_df, users,
                                                        num_negatives,
                                                        interactions_matrix,
                                                        num_items)
            utils.create_path_to_file(fpath)
            with open(fpath, 'wb') as f:
                pickle.dump(negatives, file=f)
        else:
            with open(fpath, 'rb') as f:
                negatives = pickle.load(f)
                pass
        # print(negatives)
        negatives_df = pd.DataFrame(
            negatives, columns=['user_id', 'item_id', 'target', 'day',
                                'week']).astype(np.int32)
        test_neg_df = pd.concat([test_df, negatives_df],
                                axis=0).reset_index(drop=True)

        if not utils.file_exists(path):
            results_df = run_rec(recommender, interactions_df,
                                 interactions_matrix, train_df, test_neg_df,
                                 num_users, num_items, method)
            utils.create_path_to_file(path)
            results_df.to_csv(path, index=False)
        else:
            results_df = pd.read_csv(path)

        results_df = results_df.loc[results_df['rank'] <= 10]
        mrr = utils.eval_mrr(results_df, test_neg_df)
        mrrs.append(mrr)
        ndcg = utils.eval_ndcg(results_df, test_df)
        ndcgs.append(ndcg)
        hit = utils.eval_hits(results_df, test_df)
        hits.append(hit)
        print('ndcg', ndcg, 'mrr', mrr, 'hits', hit)

    path = f'data/metrics/mrr/{execution_id}_output.csv'
    utils.create_path_to_file(path)
    pd.DataFrame(mrrs).to_csv(path, index=None)
    print('MRRs:', mrrs)
    print('Mean MRR:', np.mean(mrrs))

    path = f'data/metrics/ndcg/{execution_id}_output.csv'
    utils.create_path_to_file(path)
    pd.DataFrame(ndcgs).to_csv(path, index=None)
    print('NDCGs:', ndcgs)
    print('Mean NDCG:', np.mean(ndcgs))

    path = f'data/metrics/hit/{execution_id}_output.csv'
    utils.create_path_to_file(path)
    pd.DataFrame(hits).to_csv(path, index=None)
    print('HITs:', hits)
    print('Mean HIT:', np.mean(hits))


def train_method(recommender, method, data):
    """TODO: Docstring for train_method.

    :function: TODO
    :returns: TODO

    """
    recommender.train(data)
    if method == 'lightgcn':
        recommender.value_function.neural_network.training = False
    pass


def exec_experiment(dataset_input_parameters, methods_search_parameters,
                    methods_create_function, num_negatives, dataset_name,
                    num_executions,best_parameters):
    dataset_input_settings = dataset.dataset_settings_factory(dataset_input_parameters)
    interactions_df = dataset.parquet_load(
        dataset_input_settings['interactions_path'])
    num_users = interactions_df.user_id.max() + 1
    num_items = interactions_df.item_id.max() + 1

    print('users', num_users, 'items', num_items)

    interactions_matrix = scipy.sparse.dok_matrix((num_users, num_items),
                                                  dtype=np.int32)
    for user, item in zip(interactions_df.user_id.to_numpy(),
                          interactions_df.item_id.to_numpy()):
        interactions_matrix[user, item] = 1

    train_df, test_df = dataset.leave_one_out(interactions_df)
    # print(train_df)

    users = test_df.user_id.to_numpy()

    scootensor = None
    if 'lightgcn' in args.m or 'stacking' in args.m:
        tmp_train_df = train_df.copy()
        tmp_train_df = tmp_train_df.loc[tmp_train_df.target > 0]
        tmp_train_df = tmp_train_df.groupby(['user_id', 'item_id'
                                            ])['target'].sum() >= 1
        tmp_train_df = tmp_train_df.reset_index()
        tmp_train_df.item_id = tmp_train_df.item_id + num_users
        tmp_train_df_2 = tmp_train_df.copy()
        scootensor = torch.sparse_coo_tensor(
            np.array([
                np.hstack(
                    [tmp_train_df.user_id.values, tmp_train_df.item_id.values]),
                np.hstack(
                    [tmp_train_df.item_id.values, tmp_train_df.user_id.values]),
            ]),
            np.hstack([
                tmp_train_df.target.values.astype('int'),
                tmp_train_df.target.values.astype('int')
            ]),
            dtype=torch.float,
            size=(num_users + num_items, num_users + num_items))
        scootensor = scootensor.coalesce()


    for method, method_search_parameters in methods_search_parameters.items():

        for method_parameters in method_search_parameters:
            print((method, method_parameters, dataset_input_parameters,
                   num_executions))
            execution_id = joblib.hash(
                (method, method_parameters, dataset_input_parameters,
                 num_executions))
            method_parameters = copy(method_parameters)
            dtmp1 = {
                'num_users': num_users,
                'num_items': num_items,
                'num_batchs': 400,
                'batch_size': int(len(train_df)*0.8),
                'scootensor': scootensor,
            }
            method_parameters.update(dtmp1)
            if method == 'stacking':
                stmodels = [methods_create_function[i](utils.dict_union(best_parameters[i],dtmp1)).value_function for i in method_parameters['models']]
                # print(stmodels)
                method_parameters.update({'models': stmodels})
                # print(
            # print(method_parameters['models'])
            # method_parameters
            recommender = methods_create_function[method](method_parameters)
            leave_one_out_experiment(recommender=recommender,
                                     dataset_name=dataset_name,
                                     execution_id=execution_id,
                                     interactions_df=interactions_df,
                                     method=method,
                                     interactions_matrix=interactions_matrix,
                                     num_executions=num_executions,
                                     num_items=num_items,
                                     num_negatives=num_negatives,
                                     num_users=num_users,
                                     test_df=test_df,
                                     train_df=train_df,
                                     users=users)


def run_rec(recommender, interactions_df, interactions_matrix, train_df,
            test_df, num_users, num_items, method):
    results = []
    groups = [group for name, group in tqdm(test_df.groupby('user_id'))]
    num_groups = len(groups)
    chunksize = num_groups // 50
    groups_chunks = list(utils.chunks(groups, chunksize))
    print("Starting recommender...")
    users_num_items_recommended_online_test = defaultdict(lambda: 1)
    for groups_tmp in tqdm(groups_chunks, total=len(groups_chunks)):
        chunk_users_items_df = pd.concat(groups_tmp, axis=0)
        # print(chunk_users_items_df)
        if method in ['contextualpopularitynet', 'stacking', 'embmlp']:
            users, items = recommender.recommend(
                chunk_users_items_df['user_id'].to_numpy(),
                chunk_users_items_df['item_id'].to_numpy(),
                chunk_users_items_df)
        else:
            users, items = recommender.recommend(
                chunk_users_items_df['user_id'].to_numpy(),
                chunk_users_items_df['item_id'].to_numpy())
        chunk_users_items_df = chunk_users_items_df.set_index('user_id')
        for user_tmp, item_tmp in zip(users, items):
            results.append([
                user_tmp, item_tmp,
                users_num_items_recommended_online_test[user_tmp]
            ])
            users_num_items_recommended_online_test[user_tmp] += 1

    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'rank'])
    return results_df


argparser = argparse.ArgumentParser()
argparser.add_argument('-m', nargs='*')
argparser.add_argument('--best', action='store_true')
argparser.add_argument('--sample', action='store_true')
argparser.add_argument('--num_executions', type=int)
args = argparser.parse_args()
best_parameters = utils.load_best_parameters()
if not args.best:
    methods_search_parameters = {
        'svd': parameters.SVD_PARAMETERS,
        'svdpp': parameters.SVDPP_PARAMETERS,
        'ncf': parameters.NCF_PARAMETERS,
        'bi': parameters.BI_PARAMETERS,
        'lightgcn': parameters.LIGHTGCN_PARAMETERS,
        'stacking': parameters.STACKING_PARAMETERS,
        # 'random': [{}],
        # 'popular': [{}],
    }
    methods_search_parameters = {
        i: methods_search_parameters[i] for i in args.m
    }

methods_create_function = {
    'svd': parameters.create_svd,
    'svdpp': parameters.create_svdpp,
    'ncf': parameters.create_ncf,
    'bi': parameters.create_bi,
    'lightgcn': parameters.create_lightgcn,
    'stacking': parameters.create_stacking,
    'popular': parameters.create_popular,
    'random': parameters.create_random,
    'stacking': parameters.create_stacking,
}

num_executions =args.num_executions
# if args.best:
    # num_executions = 5
# else:
    # num_executions = 1

# for dataset_name in ['amazon_cloth']:
for dataset_name in ['amazon_fashion', 'amazon_cloth']:
# for dataset_name in ['amazon_fashion',]:
    for mshi in [5, 10]:
        dataset_input_parameters = {dataset_name: {}}

        dataset_input_parameters = {
            'preprocess': {
                'base': dataset_input_parameters,
                'mshi': mshi
            }
        }
        if args.sample:
            dataset_input_parameters = {
                'sample': {
                    'base': dataset_input_parameters,
                }
            }

        if args.best:
            dataset_input_settings = dataset.dataset_settings_factory(
                dataset_input_parameters)
            methods_search_parameters = {
                i: [best_parameters[dataset_name][mshi][i]] for i in args.m
            }
        best_parameters_data= best_parameters[dataset_name][mshi]

        exec_experiment(dataset_input_parameters=dataset_input_parameters,
                        methods_search_parameters=methods_search_parameters,
                        methods_create_function=methods_create_function,
                        num_negatives=99,
                        dataset_name=dataset_name,
                        num_executions=num_executions,best_parameters = best_parameters_data)
