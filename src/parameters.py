from neural_networks import BilinearNet, NCF
import sklearn
import torch
import loss_functions
import recommenders
import dataset
import neural_networks
import value_functions
import math
from sklearn.experimental import enable_halving_search_cv # noqa


def create_popular(parameters):
    vf = value_functions.PopularVF()
    recommender = recommenders.SimpleRecommender(vf, name='popular')
    return recommender


def create_random(parameters):
    vf = value_functions.RandomVF()
    recommender = recommenders.SimpleRecommender(vf, name='random')
    return recommender


def create_bi(parameters):
    loss_function = loss_functions.BPRLoss(1e-3, 0.001)
    nn = neural_networks.BilinearNet(parameters['num_users'],
                                     parameters['num_items'],
                                     32,
                                     sparse=False)
    nnvf = value_functions.NNVF(nn,
                                loss_function,
                                num_batchs=800,
                                batch_size=parameters['batch_size'])
    recommender = recommenders.NNRecommender(nnvf, name='bi')
    return recommender


BI_PARAMETERS = [
    {
        'weightdecay': 1e-3,
        'lr': 0.001,
        'embedding_dim': 8,
        'num_batchs': 500
    },
    {
        'weightdecay': 1e-3,
        'lr': 0.001,
        'embedding_dim': 32,
        'num_batchs': 500
    },
    {
        'weightdecay': 1e-3,
        'lr': 0.001,
        'embedding_dim': 64,
        'num_batchs': 500
    },
    # {'weightdecay': 1e-3, 'lr':0.001, 'embedding_dim': 200, 'num_batchs': 500},
    {
        'weightdecay': 1e-3,
        'lr': 0.01,
        'embedding_dim': 8,
        'num_batchs': 500
    },
    {
        'weightdecay': 1e-3,
        'lr': 0.01,
        'embedding_dim': 32,
        'num_batchs': 500
    },
    {
        'weightdecay': 1e-3,
        'lr': 0.01,
        'embedding_dim': 64,
        'num_batchs': 500
    },
    # {'weightdecay': 1e-3, 'lr':0.01, 'embedding_dim': 200, 'num_batchs': 500},
]


def create_svd(parameters):
    #model.num_lat = parameters['num_lat']
    vf = value_functions.SVDVF(num_lat=parameters['num_lat'])
    recommender = recommenders.SimpleRecommender(vf, name='svd')
    return recommender


def create_svdpp(parameters):
    #model.num_lat = parameters['num_lat']
    vf = value_functions.SVDPPVF(num_lat=parameters['num_lat'])
    recommender = recommenders.SimpleRecommender(vf, name='svdpp')
    return recommender


SVD_PARAMETERS = [
    {
        'num_lat': 8
    },
    {
        'num_lat': 32
    },
    {
        'num_lat': 64
    },
]

SVDPP_PARAMETERS = [
    {
        'num_lat': 8
    },
    {
        'num_lat': 32
    },
    {
        'num_lat': 64
    },
]


def create_ncf(parameters):
    #model.
    nn = neural_networks.NCF(parameters['num_users'], parameters['num_items'],
                             parameters['num_lat'], 4, 0.1, 'NeuMF-end')
    nnvf = value_functions.GeneralizedNNVF(
        neural_network=nn,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam(nn.parameters(), lr=parameters['lr']),
        epochs=500,
        sample_function=lambda x: dataset.sample_fixed_size(x,
                                                            len(x) // 10),
        # sample_function=lambda x: dataset.sample_fixed_size(x,100000),
        num_negatives=1,
    )
    recommender = recommenders.NNRecommender(nnvf, name='ncf')
    return recommender


NCF_PARAMETERS = [
    {
        'num_lat': 8,
        'lr': 0.1
    },
    {
        'num_lat': 32,
        'lr': 0.1
    },
    {
        'num_lat': 128,
        'lr': 0.1
    },
    {
        'num_lat': 8,
        'lr': 0.01
    },
    {
        'num_lat': 32,
        'lr': 0.01
    },
    {
        'num_lat': 128,
        'lr': 0.01
    },
]

LIGHTGCN_PARAMETERS = [
    {
        'num_lat': 8,
        'lr': 0.001
    },
    {
        'num_lat': 32,
        'lr': 0.001
    },
    # {'num_lat':64,'lr':0.001},
    {
        'num_lat': 8,
        'lr': 0.0001
    },
    {
        'num_lat': 32,
        'lr': 0.0001
    },
    # {'num_lat':64,'lr':0.0001},
]
STACKING_PARAMETERS= [
    {'models': ['popular','lightgcn']},
    {'models': ['popular']},
    {'models': ['lightgcn']},
    # {'models': []},
    {'models': ['popular','lightgcn'],'use_context':False},
    {'models': ['popular'],'use_context':False},
    {'models': ['lightgcn'],'use_context':False},
    # {'models': [],'use_context':False},
]


def create_lightgcn(parameters):
    keep_prob = 0.9
    nn = neural_networks.LightGCN(latent_dim_rec=parameters['num_lat'],
                                  lightGCN_n_layers=2,
                                  keep_prob=keep_prob,
                                  A_split=False,
                                  pretrain=0,
                                  user_emb=None,
                                  item_emb=None,
                                  dropout=1 - keep_prob,
                                  graph=parameters['scootensor'],
                                  _num_users=parameters['num_users'],
                                  _num_items=parameters['num_items'],
                                  training=True)

    loss_function = loss_functions.BPRLoss(1e-1, parameters['lr'])
    nnvf = value_functions.NNVF(nn,
                                loss_function,
                                num_batchs=parameters['num_batchs'],
                                batch_size=parameters['batch_size'])
    recommender = recommenders.NNRecommender(nnvf, name='lightgcn')
    return recommender


def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []

    nodes_increment = (last_layer_nodes - first_layer_nodes) / (n_layers - 1)
    nodes = first_layer_nodes
    for i in range(1, n_layers + 1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment

    return layers


def create_stacking(parameters):
    meta_learner_parameters = [
        dict(hidden_layer_sizes=[
            # [100],
            # [100,20],
            [100,20,5],
            [200,20,5],
            # [20,5],
            # [50,5],
            # [30],
            # [30,30],
            # [60,60],
            # [200, 20, 5],
            # FindLayerNodesLinear(5,50,10),
            # FindLayerNodesLinear(4,40,10),
            # FindLayerNodesLinear(5,100,10),
                    [10, 10, 10],
                    [10, 8, 5],
                    [10, 5, 3],
                    [10, 10],
                    [20, 10],
                    [6, 6, 6, 6],
                    [8, 8, 8, 8],
                    [8, 6, 4, 4],
            [100,5],
            # [200,5],
            FindLayerNodesLinear(4,16,2),
            FindLayerNodesLinear(5,16,2),
            FindLayerNodesLinear(6,16,2),
            FindLayerNodesLinear(3,32,2),
            FindLayerNodesLinear(3,300,2),
            FindLayerNodesLinear(4,300,2),
            FindLayerNodesLinear(3,100,10),
            # FindLayerNodesLinear(3,100,2),
            # FindLayerNodesLinear(5,20,3),
            # FindLayerNodesLinear(5,50,3),
            # FindLayerNodesLinear(4,20,3),
            # FindLayerNodesLinear(4,50,3),
            # FindLayerNodesLinear(3,20,3),
            # FindLayerNodesLinear(3,50,3),
        ],activation=['relu']
             ),
    ]

    meta_learner = sklearn.model_selection.GridSearchCV(
        sklearn.neural_network.MLPRegressor(),
        meta_learner_parameters,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2)
    vf = value_functions.Stacking(models=parameters['models'],
                                  meta_learner=meta_learner)
    if 'use_context' in parameters:
        use_context=parameters['use_context']
    else:
        use_context=True
    recommender = recommenders.SimpleRecommender(value_function=vf,
                                                 name='stacking',use_context=use_context)
    return recommender
