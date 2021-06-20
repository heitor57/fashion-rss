from torch import nn
import re
import numpy as np
import torch


def _gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def _cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


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


def _adaptive_loss(self, users, items, ratings, n_neg_candidates=5):
    negatives = Variable(
        _gpu(
            torch.from_numpy(
                np.random.randint(0, self._num_items,
                                  (len(users), n_neg_candidates))),
            self._use_cuda))
    negative_predictions = self._net(
        users.repeat(n_neg_candidates, 1).transpose_(0, 1),
        negatives).view(-1, n_neg_candidates)

    best_negative_prediction, _ = negative_predictions.max(1)
    positive_prediction = self._net(users, items)

    return torch.mean(
        torch.clamp(best_negative_prediction - positive_prediction + 1.0, 0.0))


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        # self.loss_function.neural_network = self

        self.user_embeddings = ScaledEmbedding(num_users,
                                               embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items,
                                               embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)
        self._num_users = num_users
        self._num_items = num_items
        self._use_cuda = False

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
        neg = _gpu(
            torch.from_numpy(
                np.random.randint(0, self._num_items, tuple(users.size()))),
            self._use_cuda)

        users_emb = self.user_embeddings(users.long())
        pos_emb = self.item_embeddings(pos.long())
        neg_emb = self.item_embeddings(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2)
                              + neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LSTMNet(nn.Module):

    def __init__(self, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items,
                                               embedding_dim,
                                               sparse=sparse,
                                               padding_idx=0)
        self.item_biases = ZeroEmbedding(num_items,
                                         1,
                                         sparse=sparse,
                                         padding_idx=0)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)

    def forward(self, item_sequences, item_ids):

        target_embedding = self.item_embeddings(item_ids)
        user_representations, _ = self.lstm(
            self.item_embeddings(item_sequences))
        target_bias = self.item_biases(item_ids)

        dot = (user_representations * target_embedding).sum(2)

        return dot + target_bias


class PoolNet(nn.Module):

    def __init__(self, num_items, embedding_dim, sparse=False):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.item_embeddings = ScaledEmbedding(num_items,
                                               embedding_dim,
                                               sparse=sparse,
                                               padding_idx=0)
        self.item_biases = ZeroEmbedding(num_items,
                                         1,
                                         sparse=sparse,
                                         padding_idx=0)

    def forward(self, item_sequences, item_ids):

        target_embedding = self.item_embeddings(item_ids)
        seen_embeddings = self.item_embeddings(item_sequences)
        user_representations = torch.cumsum(seen_embeddings, 1)

        target_bias = self.item_biases(item_ids)

        dot = (user_representations * target_embedding).sum(2)

        return dot + target_bias

    def bpr_loss(self, item_sequences, pos, neg):
        # for item_sequence in items_sequences:
        pos_scores = self.forward(item_sequences, pos)
        neg_scores = self.forward(item_sequences, neg)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        return loss


class PopularityNet(nn.Module):

    def __init__(self, num_items, sparse=False):
        super().__init__()
        self._num_items = num_items
        self.item_biases = ZeroEmbedding(num_items,
                                         1,
                                         sparse=sparse,
                                         padding_idx=0)

    def forward(self,item_ids):
        target_bias = self.item_biases(item_ids)
        return target_bias.flatten()
    def bpr_loss(self,users, pos, neg):
        pos_scores=self.forward(pos)
        neg_scores=self.forward(neg)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (pos_scores.norm(2).pow(2) + neg_scores.norm(2).pow(2)) / float(len(pos))
        return loss, reg_loss

class ContextualPopularityNet(nn.Module):

    def __init__(self, num_items, items_attributes,users_columns, sparse=False):
        super().__init__()
        self._num_items = num_items
        # self.item_biases = ZeroEmbedding(num_items,
                                         # 1,
                                         # sparse=sparse,
                                         # padding_idx=0)
        
        self.items_attributes = torch.tensor(items_attributes.to_numpy())
        # input_size = self.context_size+len(items_attributes.columns)
        # if hlayers == None:
            # hlayers= 
        item_context_size = len(items_attributes.columns)
        self.users_columns = users_columns
        input_size = len(users_columns)+item_context_size
        print('input_size',input_size)
        hlayers= nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//4, input_size//8),
            nn.ReLU(),
            nn.Linear(input_size//16, input_size//32),
            nn.ReLU(),
            nn.Linear(input_size//64, 1),
            nn.ReLU()
        )
        self.hlayers = hlayers


    def forward(self, users_context, items_id):
        # print(users_context.shape)
        # print(users_context[self.users_columns])
        # print(users_context[self.users_columns].shape)
        users_context= torch.tensor(users_context[self.users_columns].to_numpy(dtype=np.float32))
        # target_bias = self.item_biases(item_ids)
        # print(users_context.shape,self.items_attributes[items_id].shape)
        # print(
        t = torch.cat((users_context,self.items_attributes[items_id]),1)
        # print(t)
        # print(t.shape)
        # print(t[0].shape)
        # return self.hlayers(t[0])
        print(t.shape)
        return self.hlayers(t)
    def bpr_loss(self,users_context, pos, neg):
        # users_context= torch.tensor(users_context[self.users_columns].to_numpy())
        pos_scores=self.forward(users_context,pos)
        neg_scores=self.forward(users_context,neg)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (pos_scores.norm(2).pow(2) + neg_scores.norm(2).pow(2)) / float(len(pos))
        return loss, reg_loss

