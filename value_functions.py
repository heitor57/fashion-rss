from scipy.sparse import data
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        raise NotImplementedError

    def predict(self, targets):
        raise NotImplementedError


class NNVF(ValueFunction):

    def __init__(self, neural_network, loss_function, num_batchs=700,batch_size=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.loss_function.neural_network = neural_network
        self.num_batchs = num_batchs
        self.batch_size = batch_size
        # self.loss_function.set_optimizer()

    def train(self, dataset_):
        print(dataset_)
        dataset_ = dataset_.loc[dataset_.is_click > 0]
        self.loss_function.set_optimizer()
        t = tqdm(range(self.num_batchs))
        if isinstance(self.neural_network, (neural_networks.PoolNet)):
            users_consumed_items = dataset_[[
                'user_id', 'product_id'
            ]].groupby('user_id').agg(lambda x: torch.tensor(np.array(list(x))))
            print(users_consumed_items)
        for _ in t:
            sampled_dataset_ = dataset.sample_fixed_size(
                dataset_, self.batch_size)
            # print(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy())))
            if isinstance(self.neural_network, (neural_networks.BilinearNet,neural_networks.LightGCN)):
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset_)))
                loss = self.loss_function.compute(
                    torch.tensor(sampled_dataset_['user_id'].to_numpy()),
                    torch.tensor(sampled_dataset_['product_id'].to_numpy()),
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
                        torch.tensor(sampled_dataset_['product_id'].to_numpy()),
                        torch.tensor(sampled_dataset_['is_click'].to_numpy())):
                    loss += self.loss_function.compute(i, j, k)
                    count += 1
                loss /= count
            elif isinstance(self.neural_network,
                            (neural_networks.PopularityNet)):
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset_)))
                loss = self.loss_function.compute(None,
                    torch.tensor(sampled_dataset_['product_id'].to_numpy()), neg)
            elif isinstance(self.neural_network,
                            (neural_networks.ContextualPopularityNet)):
                # print(
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset_)))
                loss = self.loss_function.compute(sampled_dataset_,
                    torch.tensor(sampled_dataset_['product_id'].to_numpy()), neg)
                # sampled_dataset
                # self.neural_network.

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items,users_context=None):
        users = torch.tensor(users)
        items = torch.tensor(items)
        # users = torch.tensor(users,dtype=torch.long)
        # items = torch.tensor(items,dtype=torch.long)
        if isinstance(self.neural_network,neural_networks.PopularityNet):
            v = self.neural_network.forward(items)
        elif isinstance(self.neural_network,
                        (neural_networks.ContextualPopularityNet)):
            v = self.neural_network.forward(users_context, items)
        else:
            v = self.neural_network.forward(users, items)

        # print(v)
        return v
        # raise NotImplementedError


class RandomVF(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        pass

    def predict(self, users, items):
        v = np.random.random(len(users))
        return v


class GeneralizedNNVF(ValueFunction):

    def __init__(self, neural_network, optimizer, loss_function,epochs, sample_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs=  epochs
        self.sample_function=sample_function
        # self.loss_function.set_optimizer()

    def train(self, dataset_):
        print(dataset_)
        dataset_ = dataset_.loc[dataset_.is_click > 0]

        t = tqdm(range(self.epochs))
        for _ in t:
            sampled_dataset = self.sample_function(dataset_)
            # sampled_dataset = dataset.sample_fixed_size(
                # dataset_, len(dataset_))
            user_id = torch.tensor(sampled_dataset.user_id.to_numpy()).long()
            item_id = torch.tensor(sampled_dataset.product_id.to_numpy()).long()
            is_click = torch.tensor(sampled_dataset.is_click.to_numpy()).float()
            self.neural_network.zero_grad()
            if isinstance(self.neural_network, (neural_networks.PopularityNet)):
                prediction = self.neural_network(item_id)
            if isinstance(self.neural_network, (neural_networks.ContextualPopularityNet)):
                prediction = self.neural_network(sampled_dataset,item_id)
            else:
                prediction = self.neural_network(user_id, item_id)
            loss = self.loss_function(prediction, is_click)
            loss.backward()
            self.optimizer.step()

            t.set_description(f'{loss}')
            t.refresh()


    def predict(self, users, items, users_context=None):
        users = torch.tensor(users,dtype=torch.long)
        items = torch.tensor(items,dtype=torch.long)
        if isinstance(self.neural_network, (neural_networks.PopularityNet)):
            v = self.neural_network(items)
        elif isinstance(self.neural_network,
                        (neural_networks.ContextualPopularityNet)):
            v = self.neural_network.forward(users_context, items)
        else:
            v = self.neural_network(users, items)
        return v

class PopularVF(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        
        self.items_popularity = np.zeros(len(dataset_['items_attributes']))
        train = dataset_['train']
        for user_id, product_id in tqdm(train.loc[train.is_click>0].iterrows()):
            self.items_popularity[product_id['product_id']] +=1
        # for user_id, product_id in tqdm(train.loc[train.is_click>0].groupby(['user_id','product_id']).count().reset_index()[['user_id','product_id']].iterrows()):
            # self.items_popularity[product_id['product_id']] +=1
        pass

    def predict(self, users, items):
        return self.items_popularity[items]
        # v = np.random.random(len(users))
        # return v

class SVDVF(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset_):
        train= dataset_['train']
        train['is_click'].loc[train.is_click==0] = -1
        train = train.groupby(['user_id','product_id'])['is_click'].sum().reset_index()
        train['is_click'].loc[train.is_click>0] = 1
        train['is_click'].loc[train.is_click<=0] = 0
        spm = scipy.sparse.csr_matrix((train.is_click,(train.user_id,train.product_id)),shape=(dataset_['num_users'], dataset_['num_items']),dtype=float)
        # model=sklearn.decomposition.NMF(n_components=10)
        # model.fit_transform(spm)
        u, s, vt = scipy.sparse.linalg.svds(spm,k=16)
        self.U = u
        self.V = vt.T

        # NMF(n_components=50, init='random', random_state=0, verbose=True)
        pass

    def predict(self, users, items):
        values = np.empty(len(users))
        j = 0
        for u, i in zip(users,items):
            values[j] = self.U[u] @ self.V[i]
            j+=1
        return values

class Coverage(ValueFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self, dataset_):
        train= dataset_['train']
        train = train.groupby(['user_id','product_id'])['is_click'].sum().reset_index()

        spm = scipy.sparse.csr_matrix((train.is_click,(train.user_id,train.product_id)),shape=(dataset_['num_users'], dataset_['num_items']),dtype=float)

        # list_items = [i for i in tqdm(range(spm.shape[1]), position=0, leave=True) if spm[:,i].count_nonzero() >= 1]
        # list_users = [u for u in tqdm(range(spm.shape[0]), position=0, leave=True) if spm[u,:].count_nonzero() >= 1]

        list_items = pickle.load(open("data_phase1/coverage_list_items.pk", "rb"))
        list_users = pickle.load(open("data_phase1/coverage_list_users.pk", "rb"))

        # pickle.dump(list_items, open("coverage_list_items.pk", "wb"))
        # pickle.dump(list_users, open("coverage_list_users.pk", "wb"))

        # dict_nonzero = {i: set(spm[:,i].nonzero()[0]) for i in tqdm(list_items, position=0, leave=True)}

        # pickle.dump(dict_nonzero, open("coverage_dict_nonzero.pk", "wb"))
        dict_nonzero = pickle.load(open("data_phase1/coverage_dict_nonzero.pk", "rb"))

        # coverage = [(i, len(dict_nonzero[i].intersection(set(list_users)))) for i in tqdm(list_items, position=0, leave=True)]
        # coverage.sort(key=lambda x: x[1], reverse=True)
        # self.coverage = dict(coverage)

        # pickle.dump(self.coverage, open("data_phase1/coverage.pk", "wb"))
        self.coverage = pickle.load(open("data_phase1/coverage.pk", "rb"))

    def predict(self, users, items):
        result = []
        for item in items:
            if item in self.coverage:
                result.append(item)
            else:
                result.append(-99999)

        # return [self.coverage[item] for item in items] 
        return result 

class Stacking(ValueFunction):
    
    def __init__(self, models):
        self.models = models

    def train(self, dataset_train):
       
        self.items_values = {}
        for model_name in self.models:
            
            if model_name == "NCFVF":
                
                self.items_values[model_name] = np.zeros(len(dataset_train['items_attributes']))
                self.models[model_name].train(dataset_train["train"])
                # pickle.dump(self.models[model_name], open("model_ncf.pk", "wb"))
                self.models[model_name] = pickle.load(open("model_ncf.pk", "rb"))
                for name, group in tqdm(dataset_train["train"].groupby('query_id')):
                    values = self.models[model_name].value_function.predict(group['user_id'].to_numpy(),group['product_id'].to_numpy())
                    for i, pid in enumerate(group['product_id'].to_numpy()):
                        self.items_values[model_name][pid] = values[i]

            elif model_name == "PopularVF":
                self.models[model_name].train(dataset_train)
                self.items_values[model_name] = self.models[model_name].value_function.items_popularity

        # pickle.dump(self.items_values, open("items_values.pk", "wb"))

    def predict(self, users, items):
        # result = []
        # for item in items:
        #     if item in self.coverage:
        #         result.append(item)
        #     else:
        #         result.append(-99999)

        # # return [self.coverage[item] for item in items] 
        # return result 
        pass
