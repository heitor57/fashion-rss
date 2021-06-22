from torch import nn
import numpy as np
import torch
class Recommender:
    def __init__(self, name, *args,
                 **kwargs):
        # super().__init__(*args, **kwargs)
        self.name = name

    def recommend(self,target):
        raise NotImplementedError

    def train(self,dataset):
        raise NotImplementedError
class NNRecommender(Recommender):
    def __init__(self, value_function, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.value_function = value_function

    def recommend(self, users, items, users_context=None):
        # print(users,items)
        predict_value = self.value_function.predict(users,items,users_context=users_context)
        idxs = np.argsort(predict_value.detach().numpy().flatten())[::-1]
        # print(predict_value)
        # print(idxs)
        # input()
        # print(users,items)
        return users[idxs], items[idxs]

        # self.neural_network
        # raise NotImplementedError
        # pass

    def train(self,dataset):
        self.value_function.train(dataset)
        # raise NotImplementedError

class SimpleRecommender(Recommender):
    def __init__(self, value_function, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.value_function = value_function

    def recommend(self, users, items):
        predict_value = self.value_function.predict(users,items)
        idxs = np.argsort(predict_value)[::-1]
        return users[idxs], items[idxs]

    def train(self,dataset):
        self.value_function.train(dataset)
