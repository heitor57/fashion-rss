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

    def __init__(self, neural_network, loss_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function
        self.loss_function.neural_network = neural_network
        # self.loss_function.set_optimizer()

    def train(self, dataset):
        print(dataset)
        self.loss_function.set_optimizer()
        t = tqdm(range(700))
        if isinstance(self.neural_network, (neural_networks.PoolNet)):
            users_consumed_items = dataset[[
                'user_id', 'product_id'
            ]].groupby('user_id').agg(lambda x: torch.tensor(np.array(list(x))))
            print(users_consumed_items)
        for _ in t:
            sampled_dataset = sample_methods.sample_fixed_size(
                dataset, int(2048))
            # print(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy())))
            if isinstance(self.neural_network, neural_networks.BilinearNet):
                loss = self.loss_function.compute(
                    torch.tensor(sampled_dataset.iloc[:, 0].to_numpy()),
                    torch.tensor(sampled_dataset.iloc[:, 1].to_numpy()),
                    torch.tensor(sampled_dataset.iloc[:, 2].to_numpy()),
                )
            elif isinstance(self.neural_network, (neural_networks.PoolNet)):
                items_sequences = [
                    users_consumed_items.loc[i]
                    for i in sampled_dataset['user_id'].to_numpy()
                ]
                loss = 0
                count = 0
                for i, j, k in zip(items_sequences,
                                   torch.tensor(sampled_dataset['product_id'].to_numpy()),
                                   torch.tensor(sampled_dataset['is_click'].to_numpy())):
                    loss += self.loss_function.compute(i, j, k)
                    count += 1
                loss /= count

            t.set_description(f'{loss}')
            t.refresh()

    def predict(self, users, items):
        users = torch.tensor(users)
        items = torch.tensor(items)
        v = self.neural_network.forward(users, items)
        # print(v)
        return v
        # raise NotImplementedError
