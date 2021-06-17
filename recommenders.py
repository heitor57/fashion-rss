from torch import nn
import torch
class Recommender:
    def __init__(self, name, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def recommend(self,target):
        raise NotImplementedError

    def train(self,dataset):
        raise NotImplementedError
class NNRecommender:
    def __init__(self, neural_network, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network

    def recommend(self,target):

        # self.neural_network
        raise NotImplementedError

    def train(self,dataset):
        raise NotImplementedError
