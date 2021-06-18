import torch
from tqdm import tqdm
import neural_networks_train
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
        for _ in tqdm(range(30)):
            sampled_dataset = sample_methods.sample_fixed_size(dataset, 2048)
            self.loss_function.compute(torch.tensor(sampled_dataset.iloc[:, 0].to_numpy()),
                                       torch.tensor(sampled_dataset.iloc[:, 1].to_numpy()),
                                       torch.tensor(sampled_dataset.iloc[:, 2].to_numpy()),
                                       )
        
    def predict(self, users, items):
        users=torch.tensor(users)
        items=torch.tensor(items)
        return self.neural_network.forward(users,items)
        pass
        # raise NotImplementedError
