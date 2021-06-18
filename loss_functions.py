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

    def compute(self, users, pos, neg):
        loss, reg_loss = self.neural_network.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
