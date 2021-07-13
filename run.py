from collections import defaultdict
import pandas as pd
import sklearn.ensemble
import sklearn.tree
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


def run(groups, recommender):
    # def run(groups):
    # print('running a group',group['user_id'])
    tmpg = recommender.value_function.neural_network.graph

    scootensor = torch.sparse_coo_tensor(tmpg[0],
                                         tmpg[1],
                                         dtype=torch.float,
                                         size=(num_users + num_items,
                                               num_users + num_items))
    recommender.value_function.neural_network.graph = scootensor
    results = []
    for group in groups:
        print('running a group', group['user_id'])
        if method == 'contextualpopularitynet':
            users, items = recommender.recommend(
                group['user_id'].to_numpy(),
                group['product_id'].to_numpy(),
                users_context=group[users_columns])
        else:
            users, items = recommender.recommend(group['user_id'].to_numpy(),
                                                 group['product_id'].to_numpy())
        user_id = group['user_id'].iloc[0]
        group = group.set_index('user_id')
        query_id = group['query_id'].iloc[0]

        j = 1
        for i in items:
            results.append([query_str_ids[query_id], product_str_ids[i], j])
            j += 1

    return results


# dataset_input_parameters = {'dummies':
# {
# 'base': {'farfetchfinal': {}},
# }
# }
# dataset_input_parameters = {'dummies':
# {
# 'base': {'farfetch': {}},
# }
# }
dataset_input_parameters = {
    'dummies': {
        'base': {
            'split': {
                'base': {
                    'farfetch': {}
                },
                'train_size': 0.8
            }
        }
    }
}
dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', type=str)
args = argparser.parse_args()
method = args.m

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids, query_int_ids = (
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

print('users', num_users, 'items', num_items)
if method == 'bi':

    loss_function = loss_functions.BPRLoss(1e-4, 0.001)
    nn = neural_networks.BilinearNet(num_users,
                                     num_items,
                                     constants.EMBEDDING_DIM,
                                     sparse=False)
    # nn = neural_networks.PoolNet(num_items,constants.EMBEDDING_DIM)
    # nn = neural_networks.PopularityNet(num_items)
    nnvf = value_functions.NNVF(nn, loss_function)

    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'random':
    vf = value_functions.RandomVF()
    recommender = recommenders.SimpleRecommender(vf, name=method)
elif method == 'svd':
    vf = value_functions.SVDVF(num_lat=32)
    recommender = recommenders.SimpleRecommender(vf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_users': num_users,
        'num_items': num_items,
    })
elif method == 'popular':
    vf = value_functions.PopularVF()
    recommender = recommenders.SimpleRecommender(vf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'items_attributes': attributes_df,
    })
elif method == 'spotlight':
    # nn = neural_networks.LSTMNet(num_items=num_items, embedding_dim=constants.EMBEDDING_DIM)
    nnvf = value_functions.SpotlightVF(
        ImplicitSequenceModel(loss='bpr',
                              representation='pooling',
                              embedding_dim=32,
                              n_iter=20,
                              batch_size=1024,
                              l2=0.0,
                              learning_rate=1e-2,
                              optimizer_func=None,
                              use_cuda=True,
                              sparse=False,
                              random_state=None,
                              num_negative_samples=5))
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_items': num_items,
        'num_users': num_users,
    })
elif method == 'lstm':
    nn = neural_networks.LSTMNet(num_items=num_items,
                                 embedding_dim=constants.EMBEDDING_DIM)
    nnvf = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(), lr=0.0001),
        epochs=100,
        sample_function=lambda x: dataset.sample_fixed_size(x, 5000)
        # sample_function=lambda x: dataset.sample_fixed_size(x,100000)
    )
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'popularitynet':
    # loss_function = loss_functions.BPRLoss(1e-3, 0.01)
    nn = neural_networks.PopularityNet(num_items)
    nnvf = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(), lr=0.1),
        sample_function=lambda x: dataset.sample_fixed_size(x,
                                                            len(x) // 10),
        epochs=200,
        num_negatives=10)
    # nnvf = value_functions.NNVF(nn,
    # loss_function,
    # num_batchs=1000,
    # batch_size=2048)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_users': num_users,
        'num_items': num_items
    })
elif method == 'contextualpopularitynet':
    # loss_function = loss_functions.BPRLoss(1e-4, 0.01)
    # loss_function = loss_functions.RegressionLoss()
    # items_columns = list(map(str,list(range(0,32))))
    # items_columns = [
    # 'season', 'collection', 'category_id_l1', 'category_id_l2',
    # 'category_id_l3', 'brand_id', 'season_year'
    # ]
    items_columns = [
        'season', 'collection', 'gender', 'category_id_l1', 'season_year'
    ]
    pattern = '|'.join(items_columns)
    items_columns = [c for c in attributes_df.columns if re.match(pattern, c)]
    users_columns = [
        'week',
        'week_day',
        'device_category',
        'device_platform',
        'user_tier',
        # 'user_country'
    ]
    pattern = '|'.join(users_columns)
    users_columns = [
        c for c in train_normalized_df.columns if re.match(pattern, c)
    ]
    nn = neural_networks.ContextualPopularityNet(num_items,
                                                 attributes_df[items_columns],
                                                 users_columns,
                                                 dropout=0.0,
                                                 num_layers=4)
    nnvf = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(),
                                   lr=0.005,
                                   weight_decay=0.0001),
        epochs=2000,
        # sample_function=lambda x: dataset.sample_fixed_size(x,len(x))
        sample_function=lambda x: dataset.sample_fixed_size(x, 2048),
        num_negatives=10)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_users': num_users,
        'num_items': num_items
    })
    # plot = seaborn.heatmap(nn.hlayers[1].weight.detach().numpy())
    # plot.figure.savefig('contextualpopularitynet_input_layer.png')

elif method == 'ncf':
    nn = neural_networks.NCF(num_users, num_items, constants.EMBEDDING_DIM, 4,
                             0.0, 'NeuMF-end')
    nnvf = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(), lr=0.01),
        epochs=100,
        sample_function=lambda x: dataset.sample_fixed_size(x,
                                                            len(x) // 10),
        # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
        num_negatives=100,
    )
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_users': num_users,
        'num_items': num_items,
    })
    # pickle.dump(recommender, open("data_phase1/recommender_NCF.pk", "wb"))
    # recommender = pickle.load(open("data_phase1/recommender_NCF.pk", "rb"))

elif method == 'coverage':
    vf = value_functions.Coverage()
    recommender = recommenders.SimpleRecommender(vf, name=method)
    recommender.train({
        'train': train_normalized_df,
        'num_users': num_users,
        'num_items': num_items,
    })

elif method == 'lightgcn':

    tmp_train_df = train_normalized_df.copy()
    tmp_train_df = tmp_train_df.loc[tmp_train_df.is_click > 0]
    tmp_train_df = tmp_train_df.groupby(['user_id', 'product_id'
                                        ])['is_click'].sum() >= 1
    tmp_train_df = tmp_train_df.reset_index()
    tmp_train_df.product_id = tmp_train_df.product_id + num_users
    tmp_train_df_2 = tmp_train_df.copy()
    # tmp_train_df.product_id = tmp_train_df.product_id + num_users
    scootensor = torch.sparse_coo_tensor(
        np.array([
            np.hstack(
                [tmp_train_df.user_id.values, tmp_train_df.product_id.values]),
            np.hstack(
                [tmp_train_df.product_id.values, tmp_train_df.user_id.values]),
        ]),
        np.hstack([
            tmp_train_df.is_click.values.astype('int'),
            tmp_train_df.is_click.values.astype('int')
        ]),
        dtype=torch.float,
        size=(num_users + num_items, num_users + num_items))
    scootensor = scootensor.coalesce()
    # print(scootensor)
    # print(scootensor.indices())
    keep_prob = 0.99
    nn = neural_networks.LightGCN(latent_dim_rec=32,
                                  lightGCN_n_layers=2,
                                  keep_prob=keep_prob,
                                  A_split=False,
                                  pretrain=0,
                                  user_emb=None,
                                  item_emb=None,
                                  dropout=1 - keep_prob,
                                  graph=scootensor,
                                  _num_users=num_users,
                                  _num_items=num_items,
                                  training=True)

    loss_function = loss_functions.BPRLoss(1e-1, 0.001)
    nnvf = value_functions.NNVF(nn,
                                loss_function,
                                num_batchs=200,
                                batch_size=int(1e6))
    recommender = recommenders.NNRecommender(nnvf, name=method)
    # recommender.train(train_normalized_df)
    recommender.train({
        # 'train': train_normalized_df.sample(10000),
        'train': train_normalized_df,
        'items_attributes': attributes_df,
        'num_users': num_users,
        'num_items': num_items,
    })
    nn.training = False
elif method == 'stacking':

    vf = value_functions.PopularVF()
    PopularVF = vf

    nn = neural_networks.NCF(num_users, num_items, constants.EMBEDDING_DIM, 4,
                             0.0, 'NeuMF-end')
    # NCFVF = value_functions.GeneralizedNNVF(
        # neural_network=nn,
        # loss_function=torch.nn.BCEWithLogitsLoss(),
        # optimizer=torch.optim.Adam(nn.parameters(), lr=0.01),
        # epochs=60,
        # # epochs=1,
        # sample_function=lambda x: dataset.sample_fixed_size(x,
                                                            # len(x) // 10),
        # num_negatives=10,
    # )
    NCFVF = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(), lr=0.01),
        epochs=100,
        sample_function=lambda x: dataset.sample_fixed_size(x,
                                                            len(x) // 10),
        # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
        num_negatives=100,
    )

    models = [PopularVF, NCFVF]

    meta_learner = sklearn.neural_network.MLPRegressor(
        hidden_layer_sizes=[100, 100, 100],
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=True,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000)
    vf = value_functions.Stacking(models=models, meta_learner=meta_learner)
    recommender = recommenders.SimpleRecommender(value_function=vf, name=method)

    recommender.train({
        # 'train': train_normalized_df.sample(10000),
        'train': train_normalized_df,
        'items_attributes': attributes_df,
        'num_users': num_users,
        'num_items': num_items,
    })

else:
    raise SystemError

results = []
product_str_ids = {v: k for k, v in product_int_ids.items()}
query_str_ids = {v: k for k, v in query_int_ids.items()}
# query_str_ids = {v: k for k, v in query_int_ids.items()}
test_users_query_id = test_normalized_df.groupby(
    'user_id')['query_id'].unique().to_dict()
test_users_query_id = {k: v[0] for k, v in test_users_query_id.items()}

groups = [group for name, group in tqdm(test_normalized_df.groupby('query_id'))]
# executor = ProcessPoolExecutor(max_workers=4)
num_groups = len(groups)
chunksize = num_groups // 20
groups_chunks = list(chunks(groups, chunksize))

# if method == 'lightgcn':
# g = recommender.value_function.neural_network.graph
# recommender.value_function.neural_network.graph= [g.indices(),g.values()]
# # recommender.vf.neural_network.graph = None
print("Starting recommender...")
# for i in tqdm(executor.map(run, groups_chunks, [recommender]*len(groups)),total=num_groups):
# # for i in tqdm(executor.map(run, groups_chunks),total=num_groups):
# results += i

# for name, group in tqdm(test_normalized_df.groupby('query_id')):
users_num_items_recommended_online_test = defaultdict(lambda: 1)
for groups_tmp in tqdm(groups_chunks,total=len(groups_chunks)):
    chunk_users_items_df = pd.concat(groups_tmp, axis=0)
    if method in ['contextualpopularitynet', 'stacking']:
        users, items = recommender.recommend(
            chunk_users_items_df['user_id'].to_numpy(),
            chunk_users_items_df['product_id'].to_numpy(),
            users_context=chunk_users_items_df)
    else:
        users, items = recommender.recommend(
            chunk_users_items_df['user_id'].to_numpy(),
            chunk_users_items_df['product_id'].to_numpy())
    # user_id = chunk_users_items_df['user_id'].iloc[0]
    chunk_users_items_df = chunk_users_items_df.set_index('user_id')
    # query_id = chunk_users_items_df['query_id'].iloc[0]
    # j = 1
    for user_tmp, item_tmp in zip(users, items):
        results.append([
            query_str_ids[test_users_query_id[user_tmp]],
            product_str_ids[item_tmp],
            users_num_items_recommended_online_test[user_tmp]
        ])
        # j += 1
        users_num_items_recommended_online_test[user_tmp] += 1

results_df = pd.DataFrame(results, columns=['query_id', 'product_id', 'rank'])
results_df.to_csv(
    f'data_phase1/data/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv',
    index=False)
