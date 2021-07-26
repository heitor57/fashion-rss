from torch import nn
import torch.sparse
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

    def __init__(self, num_items, embedding_dim,
                 item_embedding_layer=None, sparse=False):

        super(LSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=0,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=0)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)

    def user_representation(self, item_sequences):

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, items):

        target_embedding = (self.item_embeddings(items)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(items).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


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

    def __init__(self, num_items, items_attributes,users_columns, sparse=False,num_layers=4,dropout=0):
        super().__init__()
        self._num_items = num_items
        self.items_attributes = torch.tensor(items_attributes.to_numpy())
        item_context_size = len(items_attributes.columns)
        self.users_columns = users_columns
        input_size = len(users_columns)+item_context_size + 1
        self.factor_num = input_size
        self.dropout = dropout
        MLP_modules = []
        las = 0
        for i in range(num_layers):
                input_size = self.factor_num//(2 ** i)
                MLP_modules.append(nn.Dropout(p=self.dropout))
                MLP_modules.append(nn.Linear(input_size, input_size//2))
                # MLP_modules.append(nn.ReLU())
                MLP_modules.append(nn.ReLU())
                las = input_size//2
        self.hlayers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(las,1)
        self.final_function = nn.ReLU()
        # self.final_function = nn.Sigmoid()
        # self.hlayers = hlayers
        self.item_biases = ZeroEmbedding(num_items,
                                         1,
                                         sparse=sparse,
                                         padding_idx=0)


    def forward(self, users_context, items_id):
        users_context= torch.tensor(users_context[self.users_columns].to_numpy(dtype=np.float32))
        t = torch.cat((self.item_biases(items_id),users_context,self.items_attributes[items_id]),1)
        return self.final_function(self.predict_layer(self.hlayers(t))).flatten()
    def bpr_loss(self,users_context, pos, neg):
        # users_context= torch.tensor(users_context[self.users_columns].to_numpy())
        pos_scores=self.forward(users_context,pos)
        neg_scores=self.forward(users_context,neg)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (pos_scores.norm(2).pow(2) + neg_scores.norm(2).pow(2)) / float(len(pos))
        return loss, reg_loss


class NCF(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout, model, GMF_model=None, MLP_model=None):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""		
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model

		self.embed_user_GMF = nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, factor_num)
		self.embed_user_MLP = nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))

		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		else:
			predict_size = factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)

class LightGCN(nn.Module):
    def __init__(self,latent_dim_rec,lightGCN_n_layers,keep_prob,A_split,pretrain,user_emb,item_emb,dropout, graph,_num_users,_num_items,training):
        super().__init__()
        self.latent_dim_rec = latent_dim_rec
        self.lightGCN_n_layers= lightGCN_n_layers
        self.keep_prob = keep_prob
        self.A_split = A_split
        self.pretrain = pretrain
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.dropout = dropout
        # self.config = config
        self.graph = graph
        self._num_items = _num_items
        self._num_users = _num_users
        self.training = training
        self.__init_weight()

    def __init_weight(self):
        self.latent_dim = self.latent_dim_rec
        self.n_layers = self.lightGCN_n_layers
        self.keep_prob = self.keep_prob
        self.A_split = self.A_split
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self._num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self._num_items, embedding_dim=self.latent_dim)
        if self.pretrain == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.user_emb))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.item_emb))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.dropout})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        # print(x)
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        # print('okwoe ekwqe')
        users_emb = self.embedding_user.weight
        # print('32 321 31')
        items_emb = self.embedding_item.weight
        # print('2 1321 41')
        # print(users_emb,items_emb)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        # print('ewq kewqk')
        embs = [all_emb]
        if self.dropout:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph    
        # print('341')
        
        for layer in range(self.n_layers):
            # print('5713231')
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # print(g_droped.shape,all_emb.shape)
                # print(g_droped.dtype,all_emb.dtype)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self._num_users, self._num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        # print("ewqiewjqiweqjiqew")
        all_users, all_items = self.computer()
        # print("wjqiweqjiqew")
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        # print("w34iweqjiqew")
        inner_pro = torch.mul(users_emb, items_emb)
        # print("w356weqjiqew")
        gamma     = torch.sum(inner_pro, dim=1)
        # print("w356w99jiqew")
        return gamma


class CategoricalDnn(nn.Module):
    def __init__(self,
                 categorical_nuniques_to_dims,
                 num_numerical_features,
                 fc_layers_construction,
                 dropout_probability=0.):
        super(CategoricalDnn, self).__init__()

        self._make_embedding_layers(categorical_nuniques_to_dims)

        self._make_fc_layers(num_numerical_features, fc_layers_construction, dropout_probability)

    def _make_embedding_layers(self, nuniques_to_dims):
        """
        Define Embedding Layers of categorical variables.
         Properties:
             self.embedding_layer_list         :   List of Embedding Layers applied to categorical variable
             self.num_categorical_features     :   Total number of categorical variables before Embedding
             self.num_embedded_features        :   Total number of embedded features from categorical variables
             self.num_each_embedded_features   :   Number of embedded features for each categorical variable
        """
        self.num_categorical_features = len(nuniques_to_dims)
        self.num_embedded_features = 0
        self.num_each_embedded_features = []
        self.embedding_layer_list = nn.ModuleList()
        for nunique_to_dim in nuniques_to_dims:
            num_uniques = nunique_to_dim[0]
            target_dim = nunique_to_dim[1]
            self.embedding_layer_list.append(
                nn.Sequential(
                    nn.Embedding(num_uniques, target_dim),
                    nn.BatchNorm1d(target_dim),
                    nn.ReLU(inplace=True)
                )
            )
            self.num_embedded_features += target_dim
            self.num_each_embedded_features.append(target_dim)

    def _make_fc_layers(self, num_numerical_features, fc_layers_construction, dropout_p):
        """
        Define input layer, hidden layer and output layer of the Fully Connected Layers.
         Properties:
             self.fc_layer_list   :   List of Fully Connected Layers that take embedded categorical features and
                                      numerical variables as inputs
             self.output_layer    :   Output layer
        """
        num_input = self.num_embedded_features + num_numerical_features
        self.fc_layer_list = nn.ModuleList()
        for num_output in fc_layers_construction:
            self.fc_layer_list.append(
                nn.Sequential(
                    nn.Dropout(dropout_p) if dropout_p else nn.Sequential(),
                    nn.Linear(num_input, num_output),
                    nn.BatchNorm1d(num_output),
                    nn.ReLU(inplace=True)
                )
            )
            num_input = num_output
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_p) if dropout_p else nn.Sequential(),
            nn.Linear(num_input, 1)
        )

    def forward(self, input):
        # Split the input into categorical variables and numerical variables
        categorical_input = input[:, 0:self.num_categorical_features].long()
        numerical_input = input[:, self.num_categorical_features:]

        # Embed the categorical variables
        embedded = torch.zeros(input.size()[0], self.num_embedded_features)
        start_index = 0
        for i, emb_layer in enumerate(self.embedding_layer_list):
            gorl_index = start_index + self.num_each_embedded_features[i]
            embedded[:, start_index:gorl_index] = emb_layer(categorical_input[:, i])
            start_index = gorl_index

        # Concatenate the embedded categorical features and the numerical variables and pass it to FC layers
        out = torch.cat([embedded, numerical_input], axis=1)
        for hidden_layer in self.fc_layer_list:
            out = hidden_layer(out)
        return self.output_layer(out)
