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

import pickle

class ValueFunction:

    def __init__(self,name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if name == None:
            self.name=self._get_name()

    def train(self, dataset_):
        raise NotImplementedError

    def predict(self, targets):
        raise NotImplementedError

    def _get_name(self):
        return self.__class__.__name__


class NNVF(ValueFunction):

    def __init__(self, neural_network, loss_function, num_batchs=700,batch_size=2048, *args, **kwargs):
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
        train_df = train_df[['user_id','item_id','target']].drop_duplicates()
        interactions_matrix = scipy.sparse.dok_matrix((dataset_['num_users'],dataset_['num_items']),dtype=np.int32)
        self.loss_function.set_optimizer()
        t = tqdm(range(self.num_batchs))
        if isinstance(self.neural_network, (neural_networks.PoolNet)):
            users_consumed_items = dataset_[[
                'user_id', 'item_id'
            ]].groupby('user_id').agg(lambda x: torch.tensor(np.array(list(x))))
            print(users_consumed_items)
        for _ in t:
            sampled_dataset_ = dataset.sample_fixed_size(
                dataset_['train'], self.batch_size)

            negatives = []
            users = sampled_dataset_.user_id.to_numpy()
            for i in range(len(sampled_dataset_)):
                user =users[i]
                for _ in range(self.num_negatives):
                    while True:
                        # user = np.random.randint(dataset_['num_users'])
                        item = np.random.randint(dataset_['num_items'])
                        if (user, item) not in interactions_matrix:
                            negatives.append([user,item,0])
                            break

            negatives_df=pd.DataFrame(negatives,columns=['user_id','item_id','target']).astype(np.int32)
            # negatives_df.columns = 
            neg = torch.from_numpy(negatives_df.item_id.to_numpy())
            # sampled_dataset=pd.concat([sampled_dataset,negatives_df],axis=0)
            # print(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy())))
            if isinstance(self.neural_network, (neural_networks.BilinearNet,neural_networks.LightGCN)):
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
                items_sequences = [
                    users_consumed_items.loc[i]
                    for i in sampled_dataset_['user_id'].to_numpy()
                ]
                loss = 0
                count = 0
                for i, j, k in zip(
                        items_sequences,
                        torch.tensor(sampled_dataset_['item_id'].to_numpy()),
                        torch.tensor(sampled_dataset_['target'].to_numpy())):
                    loss += self.loss_function.compute(i, j, k)
                    count += 1
                loss /= count
            elif isinstance(self.neural_network,
                            (neural_networks.PopularityNet)):
                # neg = torch.from_numpy(
                    # np.random.randint(0, self.neural_network._num_items,
                                      # len(sampled_dataset_)))
                loss = self.loss_function.compute(None,
                    torch.tensor(sampled_dataset_['item_id'].to_numpy()), neg)
            elif isinstance(self.neural_network,
                            (neural_networks.ContextualPopularityNet)):
                # print(
                # neg = torch.from_numpy(
                    # np.random.randint(0, self.neural_network._num_items,
                                      # len(sampled_dataset_)))
                loss = self.loss_function.compute(sampled_dataset_,
                    torch.tensor(sampled_dataset_['item_id'].to_numpy()), neg)
                # sampled_dataset
                # self.neural_network.

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items,users_context=None):
        # users = torch.tensor(users)
        # items = torch.tensor(items)
        users = torch.tensor(users,dtype=torch.long)
        items = torch.tensor(items,dtype=torch.long)
        if isinstance(self.neural_network,neural_networks.PopularityNet):
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

    def predict(self, users, items, context):
        v = np.random.random(len(users))
        return v


class GeneralizedNNVF(ValueFunction):

    def __init__(self, neural_network, optimizer, loss_function,epochs, sample_function, num_negatives, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs=  epochs
        self.sample_function=sample_function
        self.num_negatives = num_negatives
        # self.loss_function.set_optimizer()

    def train(self, dataset_):
        train_df = dataset_['train']
        train_df = train_df.loc[train_df.target > 0]
        train_df = train_df[['user_id','item_id','target']].drop_duplicates()
        interactions_matrix = scipy.sparse.dok_matrix((dataset_['num_users'],dataset_['num_items']),dtype=np.int32)
        # (train_df.user_id.to_numpy(),train_df.item_id.to_numpy()),train_df.target.to_numpy()
        for user, item in zip(train_df.user_id.to_numpy(),train_df.item_id.to_numpy()):
            interactions_matrix[user,item] = 1
        if isinstance(self.neural_network, (neural_networks.LSTMNet)):
            self.users_items_sequences = train_df.groupby('user_id')['item_id'].agg(lambda x: np.array(list(x))).to_dict()
            self.users_items_sequences = {k: torch.tensor(v,dtype=torch.long) for k, v in self.users_items_sequences.items()}
            train_df = train_df[['user_id']].drop_duplicates()
            # print(list(self.users_items_sequences.values())[0])

        t = tqdm(range(self.epochs))
        for _ in t:
            sampled_dataset = self.sample_function(train_df)
            negatives = []
            users = sampled_dataset.user_id.to_numpy()
            for i in range(len(sampled_dataset)):
                user =users[i]
                for _ in range(self.num_negatives):
                    while True:
                        # user = np.random.randint(dataset_['num_users'])
                        item = np.random.randint(dataset_['num_items'])
                        if (user, item) not in interactions_matrix:
                            negatives.append([user,item,0])
                            break

            negatives_df=pd.DataFrame(negatives,columns=['user_id','item_id','target']).astype(np.int32)
            # negatives_df.columns = 
            sampled_dataset=pd.concat([sampled_dataset,negatives_df],axis=0)
                
            if not isinstance(self.neural_network,neural_networks.LSTMNet):
                users = torch.tensor(sampled_dataset.user_id.to_numpy()).long()
                items = torch.tensor(sampled_dataset.item_id.to_numpy()).long()
                target = torch.tensor(sampled_dataset.target.to_numpy()).float()
            self.neural_network.zero_grad()
            if isinstance(self.neural_network, (neural_networks.PopularityNet)):
                prediction = self.neural_network(items)
            elif isinstance(self.neural_network, (neural_networks.ContextualPopularityNet)):
                prediction = self.neural_network(sampled_dataset,items)
            elif isinstance(self.neural_network, (neural_networks.LSTMNet)):
                predictions = []
                # for name, group in sampled_dataset.groupby('user_id')['item_id'].agg(lambda x: torch.tensor(np.array(x))):
                # items_sequences=[self.users_items_sequences[i] for i in sampled_dataset.user_id]
                for i in sampled_dataset.user_id:
                    items_sequences = self.users_items_sequences[i]
                    # print(items_sequences.shape)
                    user_representation, _ = self.neural_network.user_representation(items_sequences)
                    prediction = self.neural_network(user_representation,items_sequences)
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
        users = torch.tensor(users,dtype=torch.long)
        items = torch.tensor(items,dtype=torch.long)
        if isinstance(self.neural_network, (neural_networks.PopularityNet)):
            v = self.neural_network(items)
        elif isinstance(self.neural_network,
                        (neural_networks.ContextualPopularityNet)):
            v = self.neural_network.forward(users_context, items)
        elif isinstance(self.neural_network,
                        (neural_networks.LSTMNet)):
            items_sequences=torch.tensor(np.array([self.users_items_sequences[i] for i in users]))
            v = self.neural_network(items_sequences,items)
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
        for index,row in train.loc[train.target>0].groupby('item_id').count().reset_index().iterrows():
            self.items_popularity[row['item_id']] += row['user_id']
        # for user_id, item_id in tqdm(train.loc[train.target>0].iterrows()):
            # self.items_popularity[item_id['item_id']] +=1
        # for user_id, item_id in tqdm(train.loc[train.target>0].groupby(['user_id','item_id']).count().reset_index()[['user_id','item_id']].iterrows()):
            # self.items_popularity[item_id['item_id']] +=1
        pass

    def predict(self, users, items,users_context=None):
        return self.items_popularity[items]/self.num_items
        # v = np.random.random(len(users))
        # return v

class SVDVF(ValueFunction):

    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

    def train(self, dataset_):
        train= dataset_['train']
        train['target'].loc[train.target==0] = -1
        train = train.groupby(['user_id','item_id'])['target'].sum().reset_index()
        train['target'].loc[train.target>0] = 1
        train['target'].loc[train.target<=0] = 0
        spm = scipy.sparse.csr_matrix((train.target,(train.user_id,train.item_id)),shape=(dataset_['num_users'], dataset_['num_items']),dtype=float)
        # model=sklearn.decomposition.NMF(n_components=10)
        # model.fit_transform(spm)
        u, s, vt = scipy.sparse.linalg.svds(spm,k=self.num_lat)
        self.U = s*u
        self.V = vt.T
        # self.s = s

        # NMF(n_components=50, init='random', random_state=0, verbose=True)
        pass

    def predict(self, users, items, contexts):
        values = np.empty(len(users))
        j = 0
        for u, i in zip(users,items):
            values[j] = self.U[u] @ self.V[i]
            j+=1
        return values

class SVDPPVF(ValueFunction):

    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

    def train(self, dataset_):
        train= dataset_['train']
        # train['target'].loc[train.target==0] = -1
        train = train.loc[train.target>0]
        train = train.groupby(['user_id','item_id'])['target'].sum().reset_index()
        train['target'].loc[train.target>0] = 1
        # train['target'].loc[train.target<=0] = 0
        spm = scipy.sparse.csr_matrix((train.target,(train.user_id,train.item_id)),shape=(dataset_['num_users'], dataset_['num_items']),dtype=float)
        self.train_matrix = spm
        self.svdpp = SVDPlusPlus(stop_criteria=0.0000001,init_std=0.5,iterations=2000)
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
    def predict_user_item(self,uids, items):
        # self = ctypes.cast(obj_id, ctypes.py_object).value
        # test_iids = np.where(user_consumption == 0)[0]
        p_us = []
        for user in uids:
            user_consumption = self.train_matrix[user, :].A.flatten()
            all_iids = np.nonzero(user_consumption)[0]
            p_u = self.svdpp.p[user] + np.sum(self.svdpp.y[all_iids], axis=0)
            p_us += [p_u]
        p_u=np.array(p_us)
        # print(p_u.shape,self.svdpp.q[items].shape)
        # items_scores = [
        items_scores= self.svdpp.r_mean + self.svdpp.b_u[uids] + self.svdpp.b_i[items] + np.sum(p_u * self.svdpp.q[items],axis=1)
            # for iid in test_iids
        # ]
        # top_iids = test_iids[np.argsort(items_scores)
                             # [::-1]][:self.result_list_size]
        return items_scores

    def predict(self, users, items, contexts):
        # values = np.empty(len(users))
        # test_uids = np.nonzero(np.sum(test_matrix > 0, axis=1).A.flatten())[0]
        # self_id = id(self)

        # with threadpool_limits(limits=1, user_api='blas'):
            # args = [(
                # self_id,
                # int(uid),
            # ) for uid in test_uids]
        values =self.predict_user_item(users,items)
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
        train= dataset_['train']
        train = train.groupby(['user_id','item_id'])['target'].sum().reset_index()

        spm = scipy.sparse.csr_matrix((train.target,(train.user_id,train.item_id)),shape=(dataset_['num_users'], dataset_['num_items']),dtype=float)

        list_items = [i for i in tqdm(range(spm.shape[1]), position=0, leave=True) if spm[:,i].count_nonzero() >= 1]
        list_users = [u for u in tqdm(range(spm.shape[0]), position=0, leave=True) if spm[u,:].count_nonzero() >= 1]

        # list_items = pickle.load(open("data_phase1/coverage_list_items.pk", "rb"))
        # list_users = pickle.load(open("data_phase1/coverage_list_users.pk", "rb"))

        # pickle.dump(list_items, open("coverage_list_items.pk", "wb"))
        # pickle.dump(list_users, open("coverage_list_users.pk", "wb"))

        dict_nonzero = {i: set(spm[:,i].nonzero()[0]) for i in tqdm(list_items, position=0, leave=True)}

        # pickle.dump(dict_nonzero, open("coverage_dict_nonzero.pk", "wb"))
        # dict_nonzero = pickle.load(open("data_phase1/coverage_dict_nonzero.pk", "rb"))

        coverage = [(i, len(dict_nonzero[i].intersection(set(list_users)))) for i in tqdm(list_items, position=0, leave=True)]
        coverage.sort(key=lambda x: x[1], reverse=True)
        self.coverage = dict(coverage)

        # pickle.dump(self.coverage, open("data_phase1/coverage.pk", "wb"))
        # self.coverage = pickle.load(open("data_phase1/coverage.pk", "rb"))

    def predict(self, users, items,context):
        result = []
        for item in items:
            if item in self.coverage:
                result.append(item)
            else:
                result.append(-99999)

        # return [self.coverage[item] for item in items] 
        return result 

def make_embedding_construction(trans, compression_factor, max_embedded_features):
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
    
    def __init__(self, models=None,meta_learner=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models
        self.meta_learner = meta_learner

    def train(self, dataset):
       
        # self.dataset = dataset
        train_df = dataset['train']
        train_df_models = train_df.loc[train_df['target'] == 1]
        attributes_df = dataset['items_attributes']
        predicted_models_scores = []
        for i, model in enumerate(self.models):
            print('training model',model)
            
            if isinstance(model,GeneralizedNNVF):
                model.train(dataset)
            elif isinstance(model,PopularVF):
                model.train(dataset)
            i = 0
            predicted_scores = model.predict(train_df_models['user_id'].to_numpy(),train_df_models['item_id'].to_numpy())
            predicted_models_scores.append(predicted_scores)
        predicted_models_scores = np.array(predicted_models_scores).T

        if dataset['name'] == 'farfetch':
            features = sklearn.preprocessing.StandardScaler().fit_transform(predicted_models_scores)
            train_df_original=dataset['train']
            user_features_df = pd.DataFrame()
            start_time = time.time()
            user_features_df['num_sessions'] = train_df_original.groupby('user_id')['session_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['observed_items'] =train_df_original.groupby('user_id')['item_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['items_clicked'] =train_df_original.loc[train_df_original.target==1].groupby('user_id')['item_id'].nunique()
            print("%s seconds" % (time.time() - start_time))
            start_time = time.time()
            user_features_df['mean_user_price'] =train_df_original.loc[train_df_original.target==1].groupby('user_id')['product_price'].mean()
            print("%s seconds" % (time.time() - start_time))
            print(user_features_df)
            print(user_features_df.describe())
            self.users_features = user_features_df.to_dict()
            d = {
                'items_clicked':0,
                'observed_items':0,
                'num_sessions':0,
                'mean_price':0,
                }
            self.users_features = defaultdict(lambda: copy.copy(d),self.users_features)
        # self.users_features = defaultdict(lambda: copy.copy(d),self.users_features)
            # dataset["train"][model.name] = items_values[model.name]
        
            user_features = pd.DataFrame([self.users_features[i] for i in train_df.user_id]).to_numpy()
            self.user_features_min_max_scaler= sklearn.preprocessing.MinMaxScaler()
            user_features = self.user_features_min_max_scaler.fit_transform(user_features)
        # item_features = pd.DataFrame([self.users_features[i] for i in train_df.item_id]).to_numpy()


            self.items_columns_to_dummies = [
                'season', 
                'collection',
                'gender',
                'category_id_l1','category_id_l2', 
                'season_year'
            ]
            pattern = '|'.join(self.items_columns_to_dummies)
            self.items_columns = [c for c in attributes_df.columns if re.match(pattern, c)]
            # make_embedding_construction(train_df,1/4,len())
            self.users_columns_to_dummies = [
                'device_category', 
                'device_platform',
                'user_tier',
                'user_country',
                'week', 'week_day',
            ]
            pattern = '|'.join(self.users_columns_to_dummies)
            self.users_columns = [c for c in train_df.columns if re.match(pattern, c)]
            self.attributes_df = attributes_df.sort_values('item_id')
            items_features = np.array([self.attributes_df.iloc[i][self.items_columns].to_numpy() for i in train_df.item_id])
            features = np.hstack([features,user_features])
            features = np.hstack([features,train_df[self.users_columns].to_numpy(),items_features])
        elif dataset['name'] == 'amazon_fashion':
            features = sklearn.preprocessing.StandardScaler().fit_transform(predicted_models_scores)
            def df_json_convert(df):
                df = df.fillna({i: {} for i in df.index})
                return pd.json_normalize(df)
    
            res = df_json_convert(train_df['style'])
            def filter_top_popular_features(res_filtered):
                res_filtered = res_filtered.copy()
                for i in res_filtered.columns:
                    top_count = res_filtered[i].value_counts()[:100]
                    res_filtered[i].loc[~res_filtered[i].isin(top_count.index)]= np.NAN
                return res_filtered
            new_res = filter_top_popular_features(res)
            res_dummies=pd.get_dummies(new_res[['Color:','Size:','Metal Type:','Style:','Length:']])
            pattern='|'.join(['Color:','Size:','Metal Type:','Style:','Length:'])
            columns = [i for i in res_dummies.columns if re.match(pattern,i)]
            def select_top_features(df, columns, k=32):
                selectkbest = sklearn.feature_selection.SelectKBest(
                    sklearn.feature_selection.chi2, k=k)
                selected_features_values = selectkbest.fit_transform(
                    df[columns], df['target'])
                selected_columns = [
                    v1 for v1, v2 in zip(columns, selectkbest.get_support()) if v2 == True
                ]
                return selected_columns

            top_interaction_features = select_top_features(pd.concat([res_dummies,train_df['target']],axis=1),columns,200)
            interaction_features = np.array(res_dummies[top_interaction_features])
            features = np.hstack([features,interaction_features])

        # self.lr_items = sklearn.linear_model.LogisticRegression()
        # self.meta_learner = sklearn.linear_model.LinearRegression()
        # print(features)
        meta_learner_result= self.meta_learner.fit(features, train_df["target"])
        print(pd.DataFrame(meta_learner_result.cv_results_))
        pd.DataFrame(meta_learner_result.cv_results_).to_csv('data_phase1/data/debug/search.csv')

        # print(self.meta_learner.intercept_)
        # print(self.meta_learner.coef_)
        # lr_items.intercept_ + lr_items.coef_[0] * PopularVF + lr_items.coef_[1] * NCFVF

    def predict(self, users, items, users_context=None):
        result = []
        models_values = []
        for model in self.models:
            values = model.predict(users,items)
            models_values.append(values)

        models_values = np.array(models_values).T
        features = sklearn.preprocessing.StandardScaler().fit_transform(models_values)
        # features = models_values
        # print(features)
        # print(features.shape)
        user_features = pd.DataFrame([self.users_features[i]  for i in users]).to_numpy()
        features = np.hstack([features,user_features])

        items_features = np.array([self.attributes_df.iloc[i][self.items_columns].to_numpy() for i in items])
        features = np.hstack([features, users_context[self.users_columns],items_features])
        # features = np.hstack([features,np.array(users_context[self.users_columns+self.items_columns])])
        # print(features)
        # print(features)
        result= self.meta_learner.predict(features)
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
        self.max_sequence_length= train.groupby('user_id')['item_id'].count().max()+30
        print('max_sequence_length',self.max_sequence_length)
        # print(train.columns)
        # print(train['week_day'])
        interactions = Interactions(
                train.user_id.to_numpy(),
                train.item_id.to_numpy()+1,
                timestamps=(train.week*train.week_day).to_numpy(),
                num_users=dataset['num_users'],
                num_items=dataset['num_items'],
                )
        train['time'] = train.week*train.week_day
        def _f(items,max_sequence_length):
            x = np.zeros(max_sequence_length)
            for i, item in enumerate(items):
                x[max_sequence_length-len(items)+i] = item+1
            return torch.tensor(x)
        self.users_sequences = train.sort_values(['user_id','time']).groupby('user_id')['item_id'].agg(lambda x: _f(x,self.max_sequence_length)).to_dict()
        self.users_sequences = defaultdict(lambda: np.zeros(self.max_sequence_length), self.users_sequences)
        # sequence_interactions= interactions.to_sequence(max_sequence_length=self.max_sequence_length)
        sequence_interactions= interactions.to_sequence(max_sequence_length=self.max_sequence_length,step_size=100)
        self.model.fit(sequence_interactions,verbose=True)

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
        v=self.model.predict(sequences,item_ids=items.reshape(-1,1))
        return v
