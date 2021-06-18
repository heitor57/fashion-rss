
from torch import nn
import torch
class ScaledEmbedding(nn.Embedding):

    def reset_parameters(self):
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):

    def reset_parameters(self):
        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


def _adaptive_loss(self, users, items, ratings,
    n_neg_candidates=5):
    negatives = Variable(
        _gpu(
            torch.from_numpy(
                np.random.randint(0, self._num_items,
                    (len(users), n_neg_candidates))),
            self._use_cuda)
    )
    negative_predictions = self._net(
        users.repeat(n_neg_candidates, 1).transpose_(0,1),
        negatives
        ).view(-1, n_neg_candidates)

    best_negative_prediction, _ = negative_predictions.max(1)
    positive_prediction = self._net(users, items)

    return torch.mean(torch.clamp(best_negative_prediction -
                                  positive_prediction
                                  + 1.0, 0.0))

class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        # self.loss_function.neural_network = self

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)
        item_bias = self.item_biases(item_ids).view(-1, 1)

        dot = (user_embedding * item_embedding).sum(1)
        
        return dot.flatten() + user_bias.flatten() + item_bias.flatten()

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_embeddings(users.long())
        pos_emb   = self.item_embeddings(pos.long())
        neg_emb   = self.item_embeddings(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
