import torch


class LossFunction:
    pass


class BPRLoss(LossFunction):

    def __init__(self, weight_decay,lr,neural_network=None):
        # self.model = recmodel
        self.weight_decay = weight_decay
        self.lr = lr
        self.neural_network = neural_network

    def set_optimizer(self):
        self.opt = torch.optim.Adam(self.neural_network.parameters(), lr=self.lr)

    def compute(self, users, pos, neg, **kwargs):
        # if isinstance(self.neural_network,[neural_networks.PopularityNet]):
            # pos_scores = self.neural_network.forward(pos)
            # neg_scores = self.neural_network.forward(neg)
            # pos_scores
        # print(users.shape)
        loss, reg_loss = self.neural_network.bpr_loss(users, pos, neg,**kwargs)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class RegressionLoss(LossFunction):

    def __init__(self, neural_network=None):
        # self.model = recmodel
        # self.weight_decay = weight_decay
        # self.lr = lr
        self.neural_network = neural_network

    def set_optimizer(self):
        self.opt = torch.optim.Adam(self.neural_network.parameters(), lr=self.lr)

    def compute(self, users, items,observed_ratings):
        # if isinstance(self.neural_network,[neural_networks.PopularityNet]):
            # pos_scores = self.neural_network.forward(pos)
            # neg_scores = self.neural_network.forward(neg)
            # pos_scores
        # print(users.shape)
        prediction_scores = self.neural_network.forward(users,items)
        loss = ((observed_ratings - prediction_scores)**2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
