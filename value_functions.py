import torch
import numpy as np
import neural_networks
from tqdm import tqdm
import sample_methods


class ValueFunction:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, dataset):
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

    def train(self, dataset):
        print(dataset)
        dataset = dataset.loc[dataset.is_click > 0]
        self.loss_function.set_optimizer()
        t = tqdm(range(self.num_batchs))
        if isinstance(self.neural_network, (neural_networks.PoolNet)):
            users_consumed_items = dataset[[
                'user_id', 'product_id'
            ]].groupby('user_id').agg(lambda x: torch.tensor(np.array(list(x))))
            print(users_consumed_items)
        for _ in t:
            sampled_dataset = sample_methods.sample_fixed_size(
                dataset, self.batch_size)
            # print(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy())))
            if isinstance(self.neural_network, (neural_networks.BilinearNet)):
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset)))
                loss = self.loss_function.compute(
                    torch.tensor(sampled_dataset['user_id'].to_numpy()),
                    torch.tensor(sampled_dataset['product_id'].to_numpy()),
                    neg,
                    # torch.tensor(
                        # sampled_dataset['products_sampled'].to_numpy()),
                )
            elif isinstance(self.neural_network, (neural_networks.PoolNet)):
                items_sequences = [
                    users_consumed_items.loc[i]
                    for i in sampled_dataset['user_id'].to_numpy()
                ]
                loss = 0
                count = 0
                for i, j, k in zip(
                        items_sequences,
                        torch.tensor(sampled_dataset['product_id'].to_numpy()),
                        torch.tensor(sampled_dataset['is_click'].to_numpy())):
                    loss += self.loss_function.compute(i, j, k)
                    count += 1
                loss /= count
            elif isinstance(self.neural_network,
                            (neural_networks.PopularityNet)):
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset)))
                loss = self.loss_function.compute(None,
                    torch.tensor(sampled_dataset['product_id'].to_numpy()), neg)
            elif isinstance(self.neural_network,
                            (neural_networks.ContextualPopularityNet)):
                # print(
                neg = torch.from_numpy(
                    np.random.randint(0, self.neural_network._num_items,
                                      len(sampled_dataset)))
                loss = self.loss_function.compute(sampled_dataset,
                    torch.tensor(sampled_dataset['product_id'].to_numpy()), neg)
                # sampled_dataset
                # self.neural_network.

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items,users_context=None):
        users = torch.tensor(users)
        items = torch.tensor(items)
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

    def train(self, dataset):
        pass

    def predict(self, users, items):
        v = np.random.random(len(users))
        return v


class NCFVF(ValueFunction):

    def __init__(self, neural_network, optimizer, loss_function,epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.epochs=  epochs
        # self.loss_function.set_optimizer()

    def train(self, dataset):
        print(dataset)
        # dataset = dataset.loc[dataset.is_click > 0]
        dataset['is_click'].loc[dataset.is_click==0] = -1
        dataset = dataset.groupby(['user_id','product_id'])['is_click'].sum().reset_index()
        if dataset.is_click.min()<0:
            dataset.is_click += np.abs(dataset.is_click.min())
        print(dataset)
        print(dataset.describe())
        print(dataset.is_click.value_counts().sort_index())
        # raise SystemExit

        t = tqdm(range(self.epochs))
        for _ in t:
            sampled_dataset = sample_methods.sample_fixed_size(
                dataset, len(dataset))
            user_id = torch.tensor(sampled_dataset.user_id.to_numpy())
            item_id = torch.tensor(sampled_dataset.product_id.to_numpy())
            is_click = torch.tensor(sampled_dataset.is_click.to_numpy())
            self.neural_network.zero_grad()
            prediction = self.neural_network(user_id, item_id)
            loss = self.loss_function(prediction, is_click)
            loss.backward()
            self.optimizer.step()

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items, users_context=None):
        users = torch.tensor(users)
        items = torch.tensor(items)
        v = self.neural_network(users, items)
        return v
