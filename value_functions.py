from pandas.compat.numpy.function import RESHAPE_DEFAULTS
from mf.SVDPlusPlus import SVDPlusPlus
import sklearn.utils
from sys import meta_path
import re
import time
import random
import copy
import scipy.sparse
import pandas as pd
from collections import defaultdict

try:
    import spotlight
except ModuleNotFoundError:
    pass

try:
    from spotlight.interactions import Interactions
except ModuleNotFoundError:
    pass

import sklearn.preprocessing
import sklearn
import sklearn.linear_model
import dataset
import torch
import scipy.sparse.linalg
import sklearn.decomposition
import scipy.sparse
import numpy as np
import neural_networks
from tqdm import tqdm

import pytorch_widedeep
import pytorch_widedeep.preprocessing
import pytorch_widedeep.models 
import pytorch_widedeep.metrics 
import pickle


def df_json_convert(df):
    df = df.fillna({i: {} for i in df.index})
    return pd.json_normalize(df)


def count_top_popular_features(df):
    top_counts = {}
    for i in df.columns:
        top_count = df[i].value_counts()[:100].index
        top_counts[i] = top_count
    return top_counts


def filter_top_popular_features(res_filtered, top_counts, columns):
    res_filtered = res_filtered.copy()
    for column in columns:
        if column in res_filtered.columns:
            top_count = top_counts[column]
            res_filtered[column].loc[~res_filtered[column].isin(top_count
                                                               )] = np.NAN
        # else:
        # res_filtered[column] = ''

    return res_filtered


class TopPopularFeatures():

    def fit(self, df):
        self.columns_style = [
            'Color:', 'Size:', 'Metal Type:', 'Style:', 'Length:'
        ]
        res = df_json_convert(df['style'])[self.columns_style]
        self.top_counts_features = count_top_popular_features(res)

        new_res = filter_top_popular_features(res, self.top_counts_features,
                                              self.columns_style)

        new_res = new_res.fillna('NONE')
        # print(new_res)
        self.one_hot_encoder_str = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False)

        # res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
        # res = pd.concat([new_res[['Color:','Size:','Metal Type:','Style:','Length:']],df[['day','week']]],axis=1)
        self.one_hot_encoder_str.fit(new_res)
        # print(self.one_hot_encoder_str.get_feature_names(['Color:','Size:','Metal Type:','Style:','Length:']))

        self.one_hot_encoder_int = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False)

        self.one_hot_encoder_int.fit(df[['day', 'week']])
        # print(self.one_hot_encoder_int.get_feature_names(['day','week']))
        # pattern='|'.join(['Color:','Size:','Metal Type:','Style:','Length:'])
        # self.columns_dummies = [i for i in res_dummies.columns if re.match(pattern,i)]

        # date_data = pd.get_dummies(df[['day','week']])
        # pattern='|'.join(['day','week'])
        # self.days_weeks_dummies_columns = [i for i in date_data.columns if re.match(pattern,i)]

        # new_res = filter_top_popular_features(res,self.top_counts_features)
        # # self.top_popular_features= new_res.columns
        # res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
        # pattern='|'.join(['Color:','Size:','Metal Type:','Style:','Length:'])
        # columns = [i for i in res_dummies.columns if re.match(pattern,i)]
        # interaction_features = np.array(res_dummies[columns])
        # date_data = pd.get_dummies(df[['day','week']]).to_numpy()
        # interaction_features = np.hstack([interaction_features,date_data])
        pass

    def transform(self, df):
        res = df_json_convert(df['style'])
        for i in self.columns_style:
            if i not in res.columns:
                res[i] = pd.NA
        res = res[self.columns_style]

        new_res = filter_top_popular_features(res, self.top_counts_features,
                                              self.columns_style)

        new_res = new_res.fillna('NONE')
        # print(new_res)
        # self.one_hot_encoder_str=sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')

        # res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
        # res = pd.concat([new_res[['Color:','Size:','Metal Type:','Style:','Length:']],df[['day','week']]],axis=1)
        # print(new_res.shape)
        res1 = self.one_hot_encoder_str.transform(new_res)
        # print(self.one_hot_encoder_str.get_feature_names(['Color:','Size:','Metal Type:','Style:','Length:']))

        # self.one_hot_encoder_int=sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')

        res2 = self.one_hot_encoder_int.transform(df[['day', 'week']])
        # print(type(res1))
        # print(type(res2))
        print(res1.shape, res2.shape, np.hstack([res1, res2]).shape)
        return np.hstack([res1, res2])

        # res = df_json_convert(df['style'])
        # for i in ['Color:','Size:','Metal Type:','Style:','Length:']:
        # if i not in res.columns:
        # res[i] = np.NAN
        # self.top_counts_features = count_top_popular_features(res)
        # new_res = filter_top_popular_features(res,self.top_counts_features,self.columns)
        # interaction_features = self.one_hot_encoder_str.transform(new_res)
        # self.top_popular_features= new_res.columns
        # res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
        # # pattern='|'.join(['Color:','Size:','Metal Type:','Style:','Length:'])
        # # columns = [i for i in res_dummies.columns if re.match(pattern,i)]
        # interaction_features = np.array(res_dummies[self.columns_dummies])
        # date_data = pd.get_dummies(df[['day','week']])
        # interaction_features = np.hstack([interaction_features,date_data[self.days_weeks_dummies_columns].to_numpy()])
        return interaction_features


class ValueFunction:

    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if name == None:
            self.name = self._get_name()

    def train(self, dataset_):
        raise NotImplementedError

    def predict(self, targets):
        raise NotImplementedError

    def _get_name(self):
        return self.__class__.__name__


class NNVF(ValueFunction):

    def __init__(self,
                 neural_network,
                 loss_function,
                 num_batchs=700,
                 batch_size=2048,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.loss_function.neural_network = neural_network
        self.num_batchs = num_batchs
        self.batch_size = batch_size
        self.num_negatives = 1
        # self.loss_function.set_optimizer()

    def train(self, dataset_):
        print(dataset_)
        # dataset_ = dataset_.loc[dataset_.target > 0]
        train_df = dataset_['train']
        train_df = train_df.loc[train_df.target > 0]
        train_df = train_df[['user_id', 'item_id', 'target']].drop_duplicates()
        interactions_matrix = scipy.sparse.dok_matrix(
            (dataset_['num_users'], dataset_['num_items']), dtype=np.int32)
        self.loss_function.set_optimizer()
        t = tqdm(range(self.num_batchs))
        if isinstance(self.neural_network, (neural_networks.PoolNet)):
            def _tensfarray(x):
                # print(torch.tensor(np.array(x).flatten(),dtype=torch.long))
                return torch.tensor(np.array(x).flatten(),dtype=torch.long)
            users_consumed_items = train_df[[
                'user_id', 'item_id'
            ]].groupby('user_id')['item_id'].apply(_tensfarray)
            # print(users_consumed_items)
        for _ in t:
            sampled_dataset_ = dataset.sample_fixed_size(
                dataset_['train'], self.batch_size)

            negatives = []
            users = sampled_dataset_.user_id.to_numpy()
            for i in range(len(sampled_dataset_)):
                user = users[i]
                for _ in range(self.num_negatives):
                    while True:
                        # user = np.random.randint(dataset_['num_users'])
                        item = np.random.randint(dataset_['num_items'])
                        if (user, item) not in interactions_matrix:
                            negatives.append([user, item, 0])
                            break

            negatives_df = pd.DataFrame(
                negatives, columns=['user_id', 'item_id',
                                    'target']).astype(np.int32)
            # negatives_df.columns =
            neg = torch.from_numpy(negatives_df.item_id.to_numpy())
            # sampled_dataset=pd.concat([sampled_dataset,negatives_df],axis=0)
            # print(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy())))
            if isinstance(
                    self.neural_network,
                (neural_networks.BilinearNet, neural_networks.LightGCN)):
                # neg = torch.from_numpy(
                # np.random.randint(0, self.neural_network._num_items,
                # len(sampled_dataset_)))
                loss = self.loss_function.compute(
                    torch.tensor(sampled_dataset_['user_id'].to_numpy()),
                    torch.tensor(sampled_dataset_['item_id'].to_numpy()),
                    neg,
                    # torch.tensor(
                    # sampled_dataset['products_sampled'].to_numpy()),
                )
            elif isinstance(self.neural_network, (neural_networks.PoolNet)):
                sampled_dataset_users_ids = sampled_dataset_['user_id'].unique().to_numpy()
                items_sequences = [
                    users_consumed_items.loc[i]
                    for i in sampled_dataset_users_ids
                ]
                # sampled_dataset_users_consumption=sampled_dataset_[['user_id','item_id']].groupby('user_id').apply(lambda x: torch.tensor(np.array(x),dtype=torch.long))
                loss = 0
                count = 0
                for i, j, k in zip(
                        items_sequences,
                        torch.tensor(sampled_dataset_['item_id'].to_numpy(),dtype=torch.long),
                        neg):
                    # self.loss
                    loss += self.loss_function.compute(
                            i,j,k
                            )
                    count+=1
                # loss = self.loss_function.compute(
                        # items_sequences,
                        # torch.tensor(sampled_dataset_['item_id'].to_numpy(),dtype=torch.long),
                        # neg
                        # )
                # count += 1
                loss /= count
            elif isinstance(self.neural_network,
                            (neural_networks.PopularityNet)):
                # neg = torch.from_numpy(
                # np.random.randint(0, self.neural_network._num_items,
                # len(sampled_dataset_)))
                loss = self.loss_function.compute(
                    None, torch.tensor(sampled_dataset_['item_id'].to_numpy()),
                    neg)
            elif isinstance(self.neural_network,
                            (neural_networks.ContextualPopularityNet)):
                # print(
                # neg = torch.from_numpy(
                # np.random.randint(0, self.neural_network._num_items,
                # len(sampled_dataset_)))
                loss = self.loss_function.compute(
                    sampled_dataset_,
                    torch.tensor(sampled_dataset_['item_id'].to_numpy()), neg)
                # sampled_dataset
                # self.neural_network.

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items, users_context=None):
        # users = torch.tensor(users)
        # items = torch.tensor(items)
        users = torch.tensor(users, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        if isinstance(self.neural_network, neural_networks.PopularityNet):
            v = self.neural_network.forward(items)
        elif isinstance(self.neural_network,
                        (neural_networks.ContextualPopularityNet)):
            v = self.neural_network.forward(users_context, items)
        else:
            v = self.neural_network.forward(users, items)

        # print(v)
        return v.detach().numpy().flatten()
        # raise NotImplementedError


class RandomVF(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        pass

    def predict(self, users, items, context=None):
        v = np.random.random(len(users))
        return v


class GeneralizedNNVF(ValueFunction):

    def __init__(self, neural_network, optimizer, loss_function, epochs,
                 sample_function, num_negatives, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.sample_function = sample_function
        self.num_negatives = num_negatives
        # self.loss_function.set_optimizer()

    def train(self, dataset_):
        train_df = dataset_['train']
        train_df = train_df.loc[train_df.target > 0]
        train_df = train_df[['user_id', 'item_id', 'target']].drop_duplicates()
        interactions_matrix = scipy.sparse.dok_matrix(
            (dataset_['num_users'], dataset_['num_items']), dtype=np.int32)
        # (train_df.user_id.to_numpy(),train_df.item_id.to_numpy()),train_df.target.to_numpy()
        for user, item in zip(train_df.user_id.to_numpy(),
                              train_df.item_id.to_numpy()):
            interactions_matrix[user, item] = 1
        if isinstance(self.neural_network, (neural_networks.LSTMNet)):
            self.users_items_sequences = train_df.groupby('user_id')[
                'item_id'].agg(lambda x: np.array(list(x))).to_dict()
            self.users_items_sequences = {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in self.users_items_sequences.items()
            }
            train_df = train_df[['user_id']].drop_duplicates()
            # print(list(self.users_items_sequences.values())[0])

        t = tqdm(range(self.epochs))
        for _ in t:
            sampled_dataset = self.sample_function(train_df)
            negatives = []
            users = sampled_dataset.user_id.to_numpy()
            for i in range(len(sampled_dataset)):
                user = users[i]
                for _ in range(self.num_negatives):
                    while True:
                        # user = np.random.randint(dataset_['num_users'])
                        item = np.random.randint(dataset_['num_items'])
                        if (user, item) not in interactions_matrix:
                            negatives.append([user, item, 0])
                            break

            negatives_df = pd.DataFrame(
                negatives, columns=['user_id', 'item_id',
                                    'target']).astype(np.int32)
            # negatives_df.columns =
            sampled_dataset = pd.concat([sampled_dataset, negatives_df], axis=0)

            if not isinstance(self.neural_network, neural_networks.LSTMNet):
                users = torch.tensor(sampled_dataset.user_id.to_numpy()).long()
                items = torch.tensor(sampled_dataset.item_id.to_numpy()).long()
                target = torch.tensor(sampled_dataset.target.to_numpy()).float()
            self.neural_network.zero_grad()
            if isinstance(self.neural_network, (neural_networks.PopularityNet)):
                prediction = self.neural_network(items)
            elif isinstance(self.neural_network,
                            (neural_networks.ContextualPopularityNet)):
                prediction = self.neural_network(sampled_dataset, items)
            elif isinstance(self.neural_network, (neural_networks.LSTMNet)):
                predictions = []
                # for name, group in sampled_dataset.groupby('user_id')['item_id'].agg(lambda x: torch.tensor(np.array(x))):
                # items_sequences=[self.users_items_sequences[i] for i in sampled_dataset.user_id]
                for i in sampled_dataset.user_id:
                    items_sequences = self.users_items_sequences[i]
                    # print(items_sequences.shape)
                    user_representation, _ = self.neural_network.user_representation(
                        items_sequences)
                    prediction = self.neural_network(user_representation,
                                                     items_sequences)
                    predictions.append(prediction)
                prediction = torch.tensor(predictions)
            else:
                prediction = self.neural_network(users, items)
            loss = self.loss_function(prediction, target)
            loss.backward()
            self.optimizer.step()

            t.set_description(f'{loss}')
            t.refresh()

        self.neural_network.embed_user_MLP

    def predict(self, users, items, users_context=None):
        users = torch.tensor(users, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        if isinstance(self.neural_network, (neural_networks.PopularityNet)):
            v = self.neural_network(items)
        elif isinstance(self.neural_network,
                        (neural_networks.ContextualPopularityNet)):
            v = self.neural_network.forward(users_context, items)
        elif isinstance(self.neural_network, (neural_networks.LSTMNet)):
            items_sequences = torch.tensor(
                np.array([self.users_items_sequences[i] for i in users]))
            v = self.neural_network(items_sequences, items)
        else:
            v = self.neural_network(users, items)
        return v.detach().numpy()


class PopularVF(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):

        self.items_popularity = np.zeros(dataset_['num_items'])
        train = dataset_['train']
        self.num_items = dataset_['num_items']
        for index, row in train.loc[train.target > 0].groupby(
                'item_id').count().reset_index().iterrows():
            self.items_popularity[row['item_id']] += row['user_id']
        # for user_id, item_id in tqdm(train.loc[train.target>0].iterrows()):
        # self.items_popularity[item_id['item_id']] +=1
        # for user_id, item_id in tqdm(train.loc[train.target>0].groupby(['user_id','item_id']).count().reset_index()[['user_id','item_id']].iterrows()):
        # self.items_popularity[item_id['item_id']] +=1
        pass

    def predict(self, users, items, users_context=None):
        return self.items_popularity[items] / self.num_items
        # v = np.random.random(len(users))
        # return v


class SVDVF(ValueFunction):

    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

    def train(self, dataset_):
        train = dataset_['train']
        train['target'].loc[train.target == 0] = -1
        train = train.groupby(['user_id',
                               'item_id'])['target'].sum().reset_index()
        train['target'].loc[train.target > 0] = 1
        train['target'].loc[train.target <= 0] = 0
        spm = scipy.sparse.csr_matrix(
            (train.target, (train.user_id, train.item_id)),
            shape=(dataset_['num_users'], dataset_['num_items']),
            dtype=float)
        # model=sklearn.decomposition.NMF(n_components=10)
        # model.fit_transform(spm)
        u, s, vt = scipy.sparse.linalg.svds(spm, k=self.num_lat)
        self.U = s * u
        self.V = vt.T
        # self.s = s

        # NMF(n_components=50, init='random', random_state=0, verbose=True)
        pass

    def predict(self, users, items, contexts=None):
        values = np.empty(len(users))
        j = 0
        for u, i in zip(users, items):
            values[j] = self.U[u] @ self.V[i]
            j += 1
        return values


class SVDPPVF(ValueFunction):

    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

    def train(self, dataset_):
        train = dataset_['train']
        # train['target'].loc[train.target==0] = -1
        train = train.loc[train.target > 0]
        train = train.groupby(['user_id',
                               'item_id'])['target'].sum().reset_index()
        train['target'].loc[train.target > 0] = 1
        # train['target'].loc[train.target<=0] = 0
        spm = scipy.sparse.csr_matrix(
            (train.target, (train.user_id, train.item_id)),
            shape=(dataset_['num_users'], dataset_['num_items']),
            dtype=float)
        self.train_matrix = spm
        self.svdpp = SVDPlusPlus(stop_criteria=0.0000001,
                                 init_std=0.5,
                                 iterations=2000)
        self.svdpp.fit(spm)
        # model=sklearn.decomposition.NMF(n_components=10)
        # model.fit_transform(spm)
        # u, s, vt = scipy.sparse.linalg.svds(spm,k=self.num_lat)
        # self.U = s*u
        # self.V = vt.T
        # self.s = s

        # NMF(n_components=50, init='random', random_state=0, verbose=True)
        pass

    # @staticmethod
    def predict_user_item(self, uids, items):
        # self = ctypes.cast(obj_id, ctypes.py_object).value
        # test_iids = np.where(user_consumption == 0)[0]
        p_us = []
        for user in uids:
            user_consumption = self.train_matrix[user, :].A.flatten()
            all_iids = np.nonzero(user_consumption)[0]
            p_u = self.svdpp.p[user] + np.sum(self.svdpp.y[all_iids], axis=0)
            p_us += [p_u]
        p_u = np.array(p_us)
        # print(p_u.shape,self.svdpp.q[items].shape)
        # items_scores = [
        items_scores = self.svdpp.r_mean + self.svdpp.b_u[
            uids] + self.svdpp.b_i[items] + np.sum(p_u * self.svdpp.q[items],
                                                   axis=1)
        # for iid in test_iids
        # ]
        # top_iids = test_iids[np.argsort(items_scores)
        # [::-1]][:self.result_list_size]
        return items_scores

    def predict(self, users, items, contexts=None):
        # values = np.empty(len(users))
        # test_uids = np.nonzero(np.sum(test_matrix > 0, axis=1).A.flatten())[0]
        # self_id = id(self)

        # with threadpool_limits(limits=1, user_api='blas'):
        # args = [(
        # self_id,
        # int(uid),
        # ) for uid in test_uids]
        values = self.predict_user_item(users, items)
        # print(values.shape)
        # print(values)
        # j = 0
        # for u, i in zip(users,items):
        # values[j] = self.U[u] @ self.V[i]
        # j+=1
        return values


class Coverage(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        train = dataset_['train']
        train = train.groupby(['user_id',
                               'item_id'])['target'].sum().reset_index()

        spm = scipy.sparse.csr_matrix(
            (train.target, (train.user_id, train.item_id)),
            shape=(dataset_['num_users'], dataset_['num_items']),
            dtype=float)

        list_items = [
            i for i in tqdm(range(spm.shape[1]), position=0, leave=True)
            if spm[:, i].count_nonzero() >= 1
        ]
        list_users = [
            u for u in tqdm(range(spm.shape[0]), position=0, leave=True)
            if spm[u, :].count_nonzero() >= 1
        ]

        # list_items = pickle.load(open("data_phase1/coverage_list_items.pk", "rb"))
        # list_users = pickle.load(open("data_phase1/coverage_list_users.pk", "rb"))

        # pickle.dump(list_items, open("coverage_list_items.pk", "wb"))
        # pickle.dump(list_users, open("coverage_list_users.pk", "wb"))

        dict_nonzero = {
            i: set(spm[:, i].nonzero()[0])
            for i in tqdm(list_items, position=0, leave=True)
        }

        # pickle.dump(dict_nonzero, open("coverage_dict_nonzero.pk", "wb"))
        # dict_nonzero = pickle.load(open("data_phase1/coverage_dict_nonzero.pk", "rb"))

        coverage = [(i, len(dict_nonzero[i].intersection(set(list_users))))
                    for i in tqdm(list_items, position=0, leave=True)]
        coverage.sort(key=lambda x: x[1], reverse=True)
        self.coverage = dict(coverage)

        # pickle.dump(self.coverage, open("data_phase1/coverage.pk", "wb"))
        # self.coverage = pickle.load(open("data_phase1/coverage.pk", "rb"))

    def predict(self, users, items, context=None):
        result = []
        for item in items:
            if item in self.coverage:
                result.append(item)
            else:
                result.append(-99999)

        # return [self.coverage[item] for item in items]
        return result


def make_embedding_construction(trans, compression_factor,
                                max_embedded_features):
    embedding_construction = []
    total_embedded_features = 0
    for uniques in trans.nuniques():
        target = int(uniques * compression_factor)
        if target < 2:
            target = 2
        elif target > max_embedded_features:
            target = max_embedded_features
        embedding_construction.append([uniques, target])
        total_embedded_features += target
    total_features = total_embedded_features + trans.num_numericals
    return embedding_construction


class Stacking(ValueFunction):

    def __init__(self, models=None, meta_learner=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models
        self.meta_learner = meta_learner

    def preprocess_interaction_features(self, dataset_name, interactions_df):
        if dataset_name in[ 'amazon_fashion','amazon_cloth']:
            res = df_json_convert(interactions_df['style'])
            new_res = filter_top_popular_features(res)
            res_dummies = pd.get_dummies(
                new_res[['Color:', 'Size:', 'Metal Type:', 'Style:',
                         'Length:']])
            pattern = '|'.join(
                ['Color:', 'Size:', 'Metal Type:', 'Style:', 'Length:'])
            columns = [i for i in res_dummies.columns if re.match(pattern, i)]
            interaction_features = np.array(res_dummies)
            date_data = pd.get_dummies(interactions_df[['day',
                                                        'week']]).to_numpy()
            interaction_features = np.hstack([interaction_features, date_data])
            return interaction_features

    def train(self, dataset):
        self.dataset_train = dataset
        # self.dataset = dataset
        train_df = dataset['train']
        train_df_models = train_df.loc[train_df['target'] == 1]
        # attributes_df = dataset['items_attributes']
        predicted_models_scores = []
        for i, model in enumerate(self.models):
            print('training model', model)

            model.train(dataset)
            i = 0
            predicted_scores = model.predict(
                train_df_models['user_id'].to_numpy(),
                train_df_models['item_id'].to_numpy())
            predicted_models_scores.append(predicted_scores)
        predicted_models_scores = np.array(predicted_models_scores).T

        if dataset['name'] == 'farfetch':
            features = sklearn.preprocessing.StandardScaler().fit_transform(
                predicted_models_scores)
            train_df_original = dataset['train']
            user_features_df = pd.DataFrame()
            start_time = time.time()
            user_features_df['num_sessions'] = train_df_original.groupby(
                'user_id')['session_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['observed_items'] = train_df_original.groupby(
                'user_id')['item_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['items_clicked'] = train_df_original.loc[
                train_df_original.target == 1].groupby(
                    'user_id')['item_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['mean_user_price'] = train_df_original.loc[
                train_df_original.target == 1].groupby(
                    'user_id')['product_price'].mean()
            print("%s seconds" % (time.time() - start_time))
            print(user_features_df)
            print(user_features_df.describe())
            self.users_features = user_features_df.to_dict()
            d = {
                'items_clicked': 0,
                'observed_items': 0,
                'num_sessions': 0,
                'mean_price': 0,
            }
            self.users_features = defaultdict(lambda: copy.copy(d),
                                              self.users_features)
            # self.users_features = defaultdict(lambda: copy.copy(d),self.users_features)
            # dataset["train"][model.name] = items_values[model.name]

            user_features = pd.DataFrame(
                [self.users_features[i] for i in train_df.user_id]).to_numpy()
            self.user_features_min_max_scaler = sklearn.preprocessing.MinMaxScaler(
            )
            user_features = self.user_features_min_max_scaler.fit_transform(
                user_features)
            # item_features = pd.DataFrame([self.users_features[i] for i in train_df.item_id]).to_numpy()

            self.items_columns_to_dummies = [
                'season', 'collection', 'gender', 'category_id_l1',
                'category_id_l2', 'season_year'
            ]
            pattern = '|'.join(self.items_columns_to_dummies)
            self.items_columns = [
                c for c in attributes_df.columns if re.match(pattern, c)
            ]
            # make_embedding_construction(train_df,1/4,len())
            self.users_columns_to_dummies = [
                'device_category',
                'device_platform',
                'user_tier',
                'user_country',
                'week',
                'week_day',
            ]
            pattern = '|'.join(self.users_columns_to_dummies)
            self.users_columns = [
                c for c in train_df.columns if re.match(pattern, c)
            ]
            self.attributes_df = attributes_df.sort_values('item_id')
            items_features = np.array([
                self.attributes_df.iloc[i][self.items_columns].to_numpy()
                for i in train_df.item_id
            ])
            features = np.hstack([features, user_features])
            features = np.hstack([
                features, train_df[self.users_columns].to_numpy(),
                items_features
            ])
        # elif dataset['name'] == 'amazon_fashion':
        elif dataset['name'] in[ 'amazon_fashion','amazon_cloth']:
            features = sklearn.preprocessing.StandardScaler().fit_transform(
                predicted_models_scores)
            # interaction_features = self.preprocess_interaction_features(dataset['name'],train_df)

            # res = df_json_convert(train_df['style'])
            # self.top_counts_features = count_top_popular_features(res)
            # new_res = filter_top_popular_features(res,self.top_counts_features)
            # # self.top_popular_features= new_res.columns
            # res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
            # pattern='|'.join(['Color:','Size:','Metal Type:','Style:','Length:'])
            # columns = [i for i in res_dummies.columns if re.match(pattern,i)]
            # interaction_features = np.array(res_dummies[columns])
            # date_data = pd.get_dummies(train_df[['day','week']]).to_numpy()
            self.interaction_context_transformer = TopPopularFeatures()
            self.interaction_context_transformer.fit(train_df)
            interaction_features = self.interaction_context_transformer.transform(
                train_df)
            # print(features.shape,interaction_features.shape)
            features = np.hstack([features, interaction_features])
            uf1 = train_df.loc[train_df.target > 0].groupby(
                'user_id')['item_id'].count()
            uf2 = train_df.groupby('user_id')['item_id'].count()
            self.user_features = defaultdict(
                lambda: [0, 0],
                pd.concat([uf1, uf2], axis=1).to_dict())

            features = np.hstack([
                features,
                np.array([
                    self.user_features[user_id]
                    for user_id in train_df_models.user_id
                ])
            ])

            # user_features_df['items_clicked'] =train_df_original.loc[train_df_original.target==1].groupby('user_id')['item_id'].nunique()

        # self.lr_items = sklearn.linear_model.LogisticRegression()
        # self.meta_learner = sklearn.linear_model.LinearRegression()
        # print(features)
        meta_learner_result = self.meta_learner.fit(features,
                                                    train_df["target"])
        print(pd.DataFrame(meta_learner_result.cv_results_))
        pd.DataFrame(meta_learner_result.cv_results_).to_csv(
            'data_phase1/data/debug/search.csv')

        # print(self.meta_learner.intercept_)
        # print(self.meta_learner.coef_)
        # lr_items.intercept_ + lr_items.coef_[0] * PopularVF + lr_items.coef_[1] * NCFVF

    def predict(self, users, items, interaction_context=None):
        result = []
        models_values = []
        for model in self.models:
            values = model.predict(users, items)
            models_values.append(values)

        models_values = np.array(models_values).T
        features = sklearn.preprocessing.StandardScaler().fit_transform(
            models_values)

        # interaction_features = self.preprocess_interaction_features(self.dataset_train['name'],interaction_context)
        interaction_features = self.interaction_context_transformer.transform(
            interaction_context)
        # features = models_values
        # print(features)
        # print(features.shape)
        # user_features = pd.DataFrame([self.users_features[i] for i in users]).to_numpy()

        user_features = np.array([self.user_features[user] for user in users])
        features = np.hstack([features, interaction_features, user_features])

        # items_features = np.array([self.attributes_df.iloc[i][self.items_columns].to_numpy() for i in items])
        # features = np.hstack([features, interaction_context[self.users_columns],items_features])
        # features = np.hstack([features,np.array(users_context[self.users_columns+self.items_columns])])
        # print(features)
        # print(features)
        result = self.meta_learner.predict(features)
        # print(result)
        return result


class SpotlightVF(ValueFunction):

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        # self.loss_function.set_optimizer()

    def train(self, dataset):
        train = dataset['train'].copy()
        train = train.loc[train.target > 0]
        self.max_sequence_length = train.groupby(
            'user_id')['item_id'].count().max() + 30
        print('max_sequence_length', self.max_sequence_length)
        # print(train.columns)
        # print(train['week_day'])
        interactions = Interactions(
            train.user_id.to_numpy(),
            train.item_id.to_numpy() + 1,
            timestamps=(train.week * train.week_day).to_numpy(),
            num_users=dataset['num_users'],
            num_items=dataset['num_items'],
        )
        train['time'] = train.week * train.week_day

        def _f(items, max_sequence_length):
            x = np.zeros(max_sequence_length)
            for i, item in enumerate(items):
                x[max_sequence_length - len(items) + i] = item + 1
            return torch.tensor(x)

        self.users_sequences = train.sort_values(
            ['user_id', 'time']).groupby('user_id')['item_id'].agg(
                lambda x: _f(x, self.max_sequence_length)).to_dict()
        self.users_sequences = defaultdict(
            lambda: np.zeros(self.max_sequence_length), self.users_sequences)
        # sequence_interactions= interactions.to_sequence(max_sequence_length=self.max_sequence_length)
        sequence_interactions = interactions.to_sequence(
            max_sequence_length=self.max_sequence_length, step_size=100)
        self.model.fit(sequence_interactions, verbose=True)

    def predict(self, users, items, users_context=None):
        # items = torch.tensor(items,dtype=torch.long)+1
        # users = torch.tensor(users,dtype=torch.long)
        # t= [self.users_sequences[u].transpose(-1,0) for u in users]
        user = users[0]
        # sequences = torch.cat(t)
        sequences = self.users_sequences[user]
        # print(sequences)
        # print(self.users_sequences[user])
        # print(self.users_sequences[user].shape)
        # x = np.zeros(self.max_sequence_length)
        # for i, item in enumerate(items):
        # x[self.max_sequence_length-len(items)+i] = item+1
        # v=self.model.predict(sequences ,item_ids=items+1)
        # print(items)
        v = self.model.predict(sequences, item_ids=items.reshape(-1, 1))
        return v


class WideAndDeep(ValueFunction):

    def __init__(self, models=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models = models
        # self.meta_learner = meta_learner

    def train(self, dataset):
        self.dataset_train = dataset
        # self.dataset = dataset
        train_df = dataset['train']
        train_df_models = train_df.loc[train_df['target'] == 1]
        # attributes_df = dataset['items_attributes']
        predicted_models_scores = []
        for i, model in enumerate(self._models):
            print('training model', model)

            model.train(dataset)
            i = 0
            predicted_scores = model.predict(
                train_df_models['user_id'].to_numpy(),
                train_df_models['item_id'].to_numpy())
            predicted_models_scores.append(predicted_scores)
        predicted_models_scores = np.array(predicted_models_scores).T
        predicted_models_scores_df = pd.DataFrame(predicted_models_scores,columns=[f'm{i}' for i in range(len(self._models))])
        predicted_models_scores_columns = list(predicted_models_scores_df.columns)


        self.columns_style = [
            'Color:', 'Size:', 'Metal Type:', 'Style:', 'Length:'
        ]
        res = df_json_convert(train_df_models['style'])[self.columns_style]
        for column in self.columns_style:
            if column not in res.columns:
                res[column] = pd.NA
        res = res[self.columns_style]
        df_train = pd.concat([train_df_models.reset_index(drop=True),res],axis=1)
        df_train = pd.concat([df_train,predicted_models_scores_df],axis=1)
        df_train[self.columns_style]=df_train[self.columns_style].fillna('NONE')
        # print(df_train.isnull().sum() )
        # print(df_train.info(verbose=True, null_counts=True) )
        # print(df_train.shape, train_df.shape)
        # print(train_df_models)
        # print(res)


        wide_cols = [
            "day",
            "week",
        ] \
                + predicted_models_scores_columns
                # + self.columns_style \
        cross_cols = [("day", "week")]
        embed_cols = [
            ("day", 3),
            ("week", 8),
            ("Color:", 16),
            ("Metal Type:", 16),
            ("Size:", 16),
            ("Length:", 16),
            ("Style:", 16),
        ]
        cont_cols = ["timestamp"] + predicted_models_scores_columns
        target_col = "target"
        # target
        target = df_train[target_col].values

        # wide
        self._wide_preprocessor = pytorch_widedeep.preprocessing.WidePreprocessor(wide_cols=wide_cols, crossed_cols=cross_cols)
        X_wide = self._wide_preprocessor.fit_transform(df_train)
        wide = pytorch_widedeep.models.Wide(wide_dim=np.unique(X_wide).shape[0], pred_dim=1)

        # deeptabular
        self._tab_preprocessor = pytorch_widedeep.preprocessing.TabPreprocessor(embed_cols=embed_cols, continuous_cols=cont_cols)
        X_tab = self._tab_preprocessor.fit_transform(df_train)
        deeptabular = pytorch_widedeep.models.TabMlp(
            mlp_hidden_dims=[30, 30],
            column_idx=self._tab_preprocessor.column_idx,
            embed_input=self._tab_preprocessor.embeddings_input,
            continuous_cols=cont_cols,
        )

        # wide and deep
        self._meta_learner = pytorch_widedeep.models.WideDeep(wide=wide, deeptabular=deeptabular)

        # train the model
        self._trainer = pytorch_widedeep.Trainer(self._meta_learner, objective="regression")
        self._trainer.fit(
            X_wide=X_wide,
            X_tab=X_tab,
            target=target,
            n_epochs=100,
            batch_size=100000,
            val_split=0.1,
        )


        # meta_learner_result= self.meta_learner.fit(features, train_df["target"])
        # print(pd.DataFrame(meta_learner_result.cv_results_))
        # pd.DataFrame(meta_learner_result.cv_results_).to_csv('data_phase1/data/debug/search.csv')

    def predict(self, users, items, interaction_context=None):
        result = []
        predicted_models_scores = []
        for model in self._models:
            values = model.predict(users, items)
            predicted_models_scores.append(values)
        predicted_models_scores = np.array(predicted_models_scores).T
        predicted_models_scores_df = pd.DataFrame(predicted_models_scores,columns=[f'm{i}' for i in range(len(self._models))])

        interaction_context=interaction_context.reset_index(drop=True)
        # predicted_models_scores_df = pd.DataFrame(models_values,columns=[f'm{i}' for i in range(len(self._models))])
        self.columns_style = [
            'Color:', 'Size:', 'Metal Type:', 'Style:', 'Length:'
        ]
        res = df_json_convert(interaction_context['style'])
        for column in self.columns_style:
            if column not in res.columns:
                res[column] = pd.NA
        res = res[self.columns_style]
        # 1 print(res)
        # print(interaction_context)
        # print(res)
        # print(predicted_models_scores_df)
        df_train = pd.concat([interaction_context,res],axis=1)
        df_train = pd.concat([df_train,predicted_models_scores_df],axis=1)
        df_train[self.columns_style]=df_train[self.columns_style].fillna('NONE')
        # print(df_train[self.columns_style])
        # df_train = pd.concat([df_train,predicted_models_scores_df],axis=1)
        X_wide = self._wide_preprocessor.fit_transform(df_train)
        X_tab = self._tab_preprocessor.fit_transform(df_train)

        # models_values = np.array(models_values).T
        # features = sklearn.preprocessing.StandardScaler().fit_transform(
            # models_values)

        # interaction_features = self.interaction_context_transformer.transform(
            # interaction_context)
        # features = np.hstack([features, interaction_features])
        result = self._trainer.predict(
            X_wide=X_wide,
            X_tab=X_tab,
            )
        # print(result,len(result),len(users))
        return result
