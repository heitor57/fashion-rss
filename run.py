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
import pickle

# dataset_input_parameters = {'dummies': 
        # {
        # 'base': {'farfetch': {}},
# }
# }
dataset_input_parameters = {'dummies': {'base': {
    'split': {
        'base': {'farfetch': {}},
        'train_size': 0.8
    }
}}}
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

print('users',num_users,'items',num_items)
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
    vf = value_functions.SVDVF()
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
elif method == 'popularitynet':
    # loss_function = loss_functions.BPRLoss(1e-3, 0.01)
    nn = neural_networks.PopularityNet(num_items)
    nnvf = value_functions.GeneralizedNNVF(neural_network=nn,
                                 loss_function=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.Adam(nn.parameters(),
                                                            lr=0.1),
                                 epochs=200)
    # nnvf = value_functions.NNVF(nn,
                                # loss_function,
                                # num_batchs=1000,
                                # batch_size=2048)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'contextualpopularitynet':
    loss_function = loss_functions.BPRLoss(1e-4, 0.01)
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
                                                 users_columns,dropout=0.0,num_layers=4)
    # nnvf = value_functions.NNVF(nn,
                                # loss_function,
                                # num_batchs=2000,
                                # batch_size=2048)
    nnvf = value_functions.GeneralizedNNVF(neural_network=nn,
                                 loss_function=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.Adam(nn.parameters(),
                                                            lr=0.0001),
                                 epochs=20)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
elif method == 'ncf':
    nn = neural_networks.NCF(num_users, num_items, constants.EMBEDDING_DIM, 3,
                             0.0, 'NeuMF-end')
    nnvf = value_functions.GeneralizedNNVF(neural_network=nn,
                                 loss_function=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.Adam(nn.parameters(),
                                                            lr=0.0001),
                                 epochs=20)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
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
                                        ])['is_click'].sum() > 1
    tmp_train_df = tmp_train_df.reset_index()
    tmp_train_df.product_id = tmp_train_df.product_id + num_users
    tmp_train_df_2 = tmp_train_df.copy()
    # tmp_train_df.product_id = tmp_train_df.product_id + num_users
    scootensor = torch.sparse_coo_tensor(
        np.array([
            np.hstack([tmp_train_df.user_id.values,tmp_train_df.product_id.values]), 
            np.hstack([tmp_train_df.product_id.values,tmp_train_df.user_id.values]), 
            ]),
        
        np.hstack([tmp_train_df.is_click.values,tmp_train_df.is_click.values])
        ,dtype=torch.float,size=(num_users+num_items,num_users+num_items))
    scootensor = scootensor.coalesce()
    nn=neural_networks.LightGCN(latent_dim_rec=50, lightGCN_n_layers=3, keep_prob=0.99, A_split=False, pretrain=0,user_emb=None,item_emb=None, dropout=0.01,graph=scootensor,_num_users=num_users,_num_items=num_items, training=True)

    loss_function = loss_functions.BPRLoss(1e-4, 0.001)
    nnvf = value_functions.NNVF(nn, loss_function,num_batchs=200)
    recommender = recommenders.NNRecommender(nnvf, name=method)
    recommender.train(train_normalized_df)
    nn.training = False
elif method == 'stacking':

    vf = value_functions.PopularVF()
    recommender_PopularVF = recommenders.SimpleRecommender(vf, name=method)

    nn = neural_networks.NCF(num_users, num_items, constants.EMBEDDING_DIM, 3,
                             0.0, 'NeuMF-end')
    nnvf = value_functions.GeneralizedNNVF(neural_network=nn,
                                 loss_function=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=torch.optim.Adam(nn.parameters(),
                                                            lr=0.0001),
                                 epochs=20)

    recommender_NCF = recommenders.NNRecommender(nnvf, name=method)
    
    models = {'PopularVF': recommender_PopularVF, 'NCFVF': recommender_NCF}

    recommender = value_functions.Stacking(models=models)
    recommender.train({
        'train': train_normalized_df,
        'items_attributes': attributes_df,
    })

else:
    raise SystemError

results = []
product_str_ids = {v: k for k, v in product_int_ids.items()}
query_str_ids = {v: k for k, v in query_int_ids.items()}
input("inicio")
for name, group in tqdm(test_normalized_df.groupby('query_id')):
    user_teste = group['user_id'].to_numpy()[0]
    print(group['user_id'].to_numpy()[0], group['product_id'].to_numpy())
    input(">>")
    # print(group['user_id'].to_numpy())
    # print(group['product_id'].to_numpy())
    if method == 'contextualpopularitynet':
        users, items = recommender.recommend(group['user_id'].to_numpy(),
                                             group['product_id'].to_numpy(),
                                             users_context=group[users_columns])
    else:
        users, items = recommender.recommend(group['user_id'].to_numpy(),
                                             group['product_id'].to_numpy())
    user_id = group['user_id'].iloc[0]
    group = group.set_index('user_id')
    query_id = group['query_id'].iloc[0]
    j = 1
    # print(items)
    for i in items:
        # print(i)
        results.append([query_str_ids[query_id], product_str_ids[i], j])
        j += 1
    # recommender.recommend()

results_df = pd.DataFrame(results, columns=['query_id', 'product_id', 'rank'])
results_df.to_csv(
    f'data_phase1/data/{dataset.get_dataset_id(dataset_input_parameters)}_{method}_output.csv',
    index=False)
