import pandas as pd
import value_functions
import os.path
import constants
import neural_networks
import loss_functions
import recommenders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import dataset
import utils

parameters = {'rate': constants.RATE, 'random_seed': constants.RANDOM_SEED}
parameters_id = utils.parameters_to_str(parameters)

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.farfetch_train_test_normalization(
dataset.parquet_load(file_name=f'data_phase1/data/train_{parameters_id}.parquet'),
dataset.parquet_load(file_name='data_phase1/validation.parquet'),
dataset.parquet_load(file_name='data_phase1/attributes.parquet')
)

if os.path.isfile(constants.negative_samples_file):
    train_normalized_df = dataset.parquet_load(constants.negative_samples_file)
else:
    train_normalized_df = dataset.negative_samples(train_normalized_df)
    dataset.parquet_save(train_normalized_df,constants.negative_samples_file)
num_users=len(user_int_ids)
num_items=len(product_int_ids)
loss_function=loss_functions.BPRLoss(1e-4,0.001)
nn = neural_networks.BilinearNet(num_users,num_items,constants.EMBEDDING_DIM,sparse=False)
nnvf = value_functions.NNVF(nn,loss_function)

recommender = recommenders.NNRecommender(nnvf,name="NN")
recommender.train(train_normalized_df)



