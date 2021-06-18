from torch import nn
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

    def recommend(self,target):

        # self.neural_network
        raise NotImplementedError
        # pass

    def train(self,dataset):
        self.value_function.train(dataset)
        # raise NotImplementedError
