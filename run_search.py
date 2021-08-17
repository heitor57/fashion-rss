dataset_name = 'amazon_fashion'
import parameters
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


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train_method(recommender, method, data):
    """TODO: Docstring for train_method.

    :function: TODO
    :returns: TODO

    """
    recommender.train(data)
    if method == 'lightgcn':
        recommender.value_function.neural_network.training = False
    pass


def run_rec(recommender, interactions_df, interactions_matrix, train_df,
            test_df, num_users, num_items):
    results = []
    # test_users_query_id = test_df.groupby(
    # 'user_id')['query_id'].unique().to_dict()
    # test_users_query_id = {k: v[0] for k, v in test_users_query_id.items()}

    groups = [group for name, group in tqdm(test_df.groupby('user_id'))]
    num_groups = len(groups)
    chunksize = num_groups // 50
    groups_chunks = list(chunks(groups, chunksize))
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
        # user_id = chunk_users_items_df['user_id'].iloc[0]
        chunk_users_items_df = chunk_users_items_df.set_index('user_id')
        # query_id = chunk_users_items_df['query_id'].iloc[0]
        # j = 1
        # print(users,items)
        for user_tmp, item_tmp in zip(users, items):
            results.append([
                user_tmp, item_tmp,
                users_num_items_recommended_online_test[user_tmp]
            ])
            users_num_items_recommended_online_test[user_tmp] += 1

    results_df = pd.DataFrame(results, columns=['user_id', 'item_id', 'rank'])
    return results_df


num_negatives = 99

dataset_input_parameters = {dataset_name: {}}

dataset_input_parameters = {
    'preprocess': {
        'base': dataset_input_parameters,
        'mshi': 5
    }
}
# dataset_input_parameters = {'sample': {'base': dataset_input_parameters}}

dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', type=list)
args = argparser.parse_args()

interactions_df = dataset.parquet_load(
    dataset_input_settings['interactions_path'])
num_users = interactions_df.user_id.max()+1
num_items = interactions_df.item_id.max()+1

print('users', num_users, 'items', num_items)

interactions_matrix = scipy.sparse.dok_matrix((num_users, num_items),
                                              dtype=np.int32)
for user, item in zip(interactions_df.user_id.to_numpy(),
                      interactions_df.item_id.to_numpy()):
    interactions_matrix[user, item] = 1

train_df, test_df = dataset.leave_one_out(interactions_df)
# print(train_df)

users = test_df.user_id.to_numpy()

num_executions = 5

for method in args.m:

    method_search_parameters= []
    create_method = lambda x: None
    if args.m == 'svd':
        method_search_parameters = parameters.SVD_PARAMETERS
        create_method = parameters.create_svd
    elif args.m == 'svdpp':
        method_search_parameters = parameters.SVDPP_PARAMETERS
        create_method = parameters.create_svdpp
    elif args.m == 'ncf':
        method_search_parameters = parameters.NCF_PARAMETERS
        create_method = parameters.create_ncf
    elif args.m == 'bi':
        method_search_parameters = parameters.BI_PARAMETERS
        create_method = parameters.create_bi
    # elif args.m == 'stacking':
        # method_search_parameters = parameters.BI_PARAMETERS
        # create_method = parameters.create_bi
    else:
        raise NameError

    for method_parameters in method_search_parameters:
        # method_parameters
        recommender = create_method(method_parameters)
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
        execution_id = joblib.hash((method,method_parameters,dataset_input_parameters,num_executions))

        for i in range(num_executions):
            seed = i
            np.random.seed(seed)
            negatives_id = joblib.hash((seed, test_df, users, num_negatives,
                                        interactions_matrix, num_items))
            fpath = f'data/utils/negatives/{negatives_id}'
            if not utils.file_exists(fpath):
                negatives = utils.generate_negative_samples(test_df, users,
                                                            num_negatives,
                                                            interactions_matrix,
                                                            num_items)
                utils.create_path_to_file(fpath)
                with open(fpath,'wb') as f:
                    pickle.dump(negatives, file=f)
            else:
                with open(fpath,'rb') as f:
                    negatives = pickle.load(f)
                    pass
            # print(negatives)

            negatives_df = pd.DataFrame(
                negatives, columns=['user_id', 'item_id', 'target', 'day',
                                    'week']).astype(np.int32)
            test_neg_df = pd.concat([test_df, negatives_df],
                                    axis=0).reset_index(drop=True)

            results_df = run_rec(recommender, interactions_df, interactions_matrix,
                                 train_df, test_neg_df, num_users, num_items)
            path = f'data/results/{execution_id}_{i}_output.csv'
            utils.create_path_to_file(path)
            results_df.to_csv(path, index=False)

            results_df = results_df.loc[results_df['rank'] <= 10]
            mrr = utils.eval_mrr(results_df, test_neg_df)
            mrrs.append(mrr)
            ndcg = utils.eval_ndcg(results_df, test_df)
            ndcgs.append(ndcg)
            hit = utils.eval_hits(results_df, test_df)
            hits.append(hit)
            print(results_df.describe())
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
