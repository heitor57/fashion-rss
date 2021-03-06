dataset_name = 'amazon_cloth'
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


def method_factory(method):
    if method == 'bi':
        if dataset_name == 'amazon_fashion':
            loss_function = loss_functions.BPRLoss(1e-3, 0.001)
            nn = neural_networks.BilinearNet(num_users, num_items, 32, sparse=False)
            # nn = neural_networks.PoolNet(num_items,constants.EMBEDDING_DIM)
            # nn = neural_networks.PopularityNet(num_items)
            nnvf = value_functions.NNVF(nn,
                                        loss_function,
                                        num_batchs=500,
                                        batch_size=int(len(train_df)*0.9))
        elif dataset_name == 'amazon_cloth':
            loss_function = loss_functions.BPRLoss(1e-3, 0.01)
            nn = neural_networks.BilinearNet(num_users, num_items, 1000, sparse=False)
            # nn = neural_networks.PoolNet(num_items,constants.EMBEDDING_DIM)
            # nn = neural_networks.PopularityNet(num_items)
            nnvf = value_functions.NNVF(nn,
                                        loss_function,
                                        num_batchs=500,
                                        batch_size=int(len(train_df)//10))

        recommender = recommenders.NNRecommender(nnvf, name=method)
    elif method == 'poolnet':
        loss_function = loss_functions.BPRLoss(1e-3, 0.001)
        nn = neural_networks.PoolNet(num_items, 32, sparse=False)
        nnvf = value_functions.NNVF(nn,
                                    loss_function,
                                    num_batchs=500,
                                    batch_size=int(len(train_df)*0.9))

        recommender = recommenders.NNRecommender(nnvf, name=method)
    elif method == 'embmlp':
        vfs= [value_functions.PopularVF(),value_functions.SVDVF(num_lat=8)]
        vf = value_functions.WideAndDeep(models=vfs)
        recommender = recommenders.SimpleRecommender(vf, name=method)
    elif method == 'random':
        vf = value_functions.RandomVF()
        recommender = recommenders.SimpleRecommender(vf, name=method)
    elif method == 'svd':
        
        if dataset_name == 'amazon_fashion':
            vf = value_functions.SVDVF(num_lat=8)
        else:
            vf = value_functions.SVDVF(num_lat=1000)
        recommender = recommenders.SimpleRecommender(vf, name=method)
    elif method == 'svdpp':
        if dataset_name == 'amazon_fashion':
            vf = value_functions.SVDPPVF(num_lat=8)
        elif dataset_name == 'amazon_cloth':
            vf = value_functions.SVDPPVF(num_lat=500)
        recommender = recommenders.SimpleRecommender(vf, name=method)
    elif method == 'popular':
        vf = value_functions.PopularVF()
        recommender = recommenders.SimpleRecommender(vf, name=method)
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
    elif method == 'popularitynet':
        # loss_function = loss_functions.BPRLoss(1e-3, 0.01)
        nn = neural_networks.PopularityNet(num_items)
        nnvf = value_functions.GeneralizedNNVF(
            neural_network=nn,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(nn.parameters(), lr=0.1),
            sample_function=lambda x: dataset.sample_fixed_size(
                x,
                len(x) // 10),
            epochs=200,
            num_negatives=10)
        # nnvf = value_functions.NNVF(nn,
        # loss_function,
        # num_batchs=1000,
        # batch_size=2048)
        recommender = recommenders.NNRecommender(nnvf, name=method)
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
        items_columns = [
            c for c in attributes_df.columns if re.match(pattern, c)
        ]
        users_columns = [
            'week',
            'week_day',
            'device_category',
            'device_platform',
            'user_tier',
            # 'user_country'
        ]
        pattern = '|'.join(users_columns)
        users_columns = [c for c in train_df.columns if re.match(pattern, c)]
        nn = neural_networks.ContextualPopularityNet(
            num_items,
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
        # plot = seaborn.heatmap(nn.hlayers[1].weight.detach().numpy())
        # plot.figure.savefig('contextualpopularitynet_input_layer.png')

    elif method == 'ncf':
        if dataset_name == 'amazon_fashion':
            nn = neural_networks.NCF(num_users, num_items, 8, 4, 0.1, 'NeuMF-end')
            nnvf = value_functions.GeneralizedNNVF(
                neural_network=nn,
                loss_function=torch.nn.BCEWithLogitsLoss(),
                optimizer=torch.optim.Adam(nn.parameters(), lr=0.01),
                epochs=400,
                sample_function=lambda x: dataset.sample_fixed_size(
                    x,
                    len(x)),
                # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
                num_negatives=1,
            )
            recommender = recommenders.NNRecommender(nnvf, name=method)
        elif dataset_name == 'amazon_cloth':
            nn = neural_networks.NCF(num_users, num_items, 128, 4, 0.1, 'NeuMF-end')
            nnvf = value_functions.GeneralizedNNVF(
                neural_network=nn,
                loss_function=torch.nn.BCEWithLogitsLoss(),
                optimizer=torch.optim.Adam(nn.parameters(), lr=0.1),
                epochs=500,
                sample_function=lambda x: dataset.sample_fixed_size(
                    x,
                    len(x)//10),
                # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
                num_negatives=1,
            )
            recommender = recommenders.NNRecommender(nnvf, name=method)

    elif method == 'coverage':
        vf = value_functions.Coverage()
        recommender = recommenders.SimpleRecommender(vf, name=method)

    elif method == 'lightgcn':

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

        if dataset_name == 'amazon_fashion':
            keep_prob = 0.9
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
                                        num_batchs=2000,
                                        batch_size=int(len(interactions_df)*0.8))
        elif dataset_name == 'amazon_cloth':
            keep_prob = 0.9
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

            loss_function = loss_functions.BPRLoss(1e-1, 0.0001)
            nnvf = value_functions.NNVF(nn,
                                        loss_function,
                                        num_batchs=400,
                                        batch_size=int(len(interactions_df)//3))

        recommender = recommenders.NNRecommender(nnvf, name=method)
        # recommender.train(train_normalized_df)
    elif method == 'stacking':
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

        vf = value_functions.PopularVF()
        PopularVF = vf
        keep_prob = 0.9
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
        LIGHTGCNVF = value_functions.NNVF(nn,
                                    loss_function,
                                    num_batchs=400,
                                    batch_size=int(len(train_df)*0.8))

        # nn = neural_networks.NCF(num_users, num_items, 8, 4, 0.1, 'NeuMF-end')
        # NCFVF = value_functions.GeneralizedNNVF(
            # neural_network=nn,
            # loss_function=torch.nn.BCEWithLogitsLoss(),
            # optimizer=torch.optim.Adam(nn.parameters(), lr=0.01),
            # epochs=2000,
            # sample_function=lambda x: dataset.sample_fixed_size(
                # x,
                # len(x)),
            # # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
            # num_negatives=1,
        # )
        # vf = 
        # models = [PopularVF, LIGHTGCNVF,NCFVF,value_functions.SVDPPVF(num_lat=8)]
        # models = [PopularVF, LIGHTGCNVF,NCFVF]
        models = [PopularVF, LIGHTGCNVF]

        meta_learner_parameters = [
            dict(
                hidden_layer_sizes=[
                    [10, 10, 10],
                    [10, 8, 5],
                    [10, 5, 3],
                    [5, 5],
                    [6, 6, 6, 6],
                    [8, 8, 8, 8],
                    [8, 6, 4, 4],
                ],),
        ]

        meta_learner = sklearn.model_selection.GridSearchCV(
            sklearn.neural_network.MLPRegressor(),
            meta_learner_parameters,
            scoring='neg_root_mean_squared_error',
            # n_jobs=-1,
            verbose=2)
        vf = value_functions.Stacking(models=models, meta_learner=meta_learner)
        recommender = recommenders.SimpleRecommender(value_function=vf,
                                                     name=method)

    else:
        raise SystemError
    return recommender


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

# dataset_input_parameters = {'amazon_fashion': {}}
dataset_input_parameters = {dataset_name: {}}

dataset_input_parameters = {'preprocess': {'base': dataset_input_parameters,'mshi': 10}}

dataset_input_settings = dataset.dataset_settings_factory(
    dataset_input_parameters)

argparser = argparse.ArgumentParser()
argparser.add_argument('-m', type=str)
args = argparser.parse_args()
method = args.m

interactions_df = dataset.parquet_load(
    dataset_input_settings['interactions_path'])
num_users = interactions_df.user_id.nunique()
num_items = interactions_df.item_id.nunique()

print('users', num_users, 'items', num_items)

interactions_matrix = scipy.sparse.dok_matrix((num_users, num_items),
                                              dtype=np.int32)
for user, item in zip(interactions_df.user_id.to_numpy(),
                      interactions_df.item_id.to_numpy()):
    interactions_matrix[user, item] = 1

train_df, test_df = dataset.leave_one_out(interactions_df)

users = test_df.user_id.to_numpy()
# print(users.dtype)
# print(test_df)
# print(test_df[['user_id','item_id']].drop_duplicates().shape,test_df.shape)

# print(test_df[['user_id','item_id']].drop_duplicates().shape,test_df.shape)
# print(test_df[['user_id', 'item_id', 'target']])

recommender = method_factory(args.m)

train_method(recommender, method, {
    'name': dataset_name,
    'train': train_df,
    'num_users': num_users,
    'num_items': num_items
})

np.random.seed(1)
mrrs = []
ndcgs=[]
hits=[]

for i in range(5):
    seed = i
    np.random.seed(seed)
    negatives_id = joblib.hash((seed,test_df,users,num_negatives,interactions_matrix,num_items))
    fpath = f'data/utils/negatives/{negatives_id}'
    if not utils.file_exists(fpath):
        negatives= utils.generate_negative_samples(test_df,users,num_negatives,interactions_matrix,num_items)
        utils.create_path_to_file(path)
        pickle.dump(negatives,file=open(fpath,'wb'))
    else:
        negatives = pickle.load(open(fpath,'rb'))

    negatives_df = pd.DataFrame(negatives, columns=['user_id', 'item_id',
                                                    'target','day','week']).astype(np.int32)
    test_neg_df = pd.concat([test_df, negatives_df], axis=0).reset_index(drop=True)

    results_df = run_rec(recommender, interactions_df, interactions_matrix,
                         train_df, test_neg_df, num_users, num_items)
    path = f'data/results/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
    utils.create_path_to_file(path)
    results_df.to_csv(path, index=False)

    results_df=results_df.loc[results_df['rank'] <= 10]
    mrr = utils.eval_mrr(results_df,test_neg_df)
    mrrs.append(mrr)
    ndcg = utils.eval_ndcg(results_df,test_df)
    ndcgs.append(ndcg)
    hit = utils.eval_hits(results_df,test_df)
    hits.append(hit)
    print('ndcg',ndcg,'mrr',mrr,'hits',hit)

path = f'data/metrics/mrr/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
utils.create_path_to_file(path)
pd.DataFrame(mrrs).to_csv(path,index=None)
print('MRRs:',mrrs)
print('Mean MRR:',np.mean(mrrs))

path = f'data/metrics/ndcg/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
utils.create_path_to_file(path)
pd.DataFrame(ndcgs).to_csv(path,index=None)
print('NDCGs:',ndcgs)
print('Mean NDCG:',np.mean(ndcgs))


path = f'data/metrics/hit/{method}_{dataset.get_dataset_id(dataset_input_parameters)}_output.csv'
utils.create_path_to_file(path)
pd.DataFrame(hits).to_csv(path,index=None)
print('HITs:',hits)
print('Mean HIT:',np.mean(hits))
