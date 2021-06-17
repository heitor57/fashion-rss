class ValueFunction:
    def __init__(self, name, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def train(self,dataset):
        raise NotImplementedError

    def predict(self,targets):
        raise NotImplementedError

class NNVF(ValueFunction):
    def __init__(self, neural_network, loss_function,  *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_network = neural_network
        self.loss_function = loss_function

    def train(self,dataset):
        raise NotImplementedError

    def predict(self,targets):
        raise NotImplementedError
