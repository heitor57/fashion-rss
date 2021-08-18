import parameters
from copy import copy
from collections import defaultdict
import scipy.sparse
import pandas as pd
import sklearn.ensemble
import sklearn.tree
import sklearn.model_selection
import sklearn.neural_network
try:
    import spotlight
    from spotlight.sequence.implicit import ImplicitSequenceModel
except ModuleNotFoundError:
    pass

import seaborn
import re
import joblib
from torch.optim import optimizer
import torch
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
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


for dataset_name in ['amazon_fashion', 'amazon_cloth']:
    for mshi in [5, 10]:
        dataset_input_parameters = {dataset_name: {}}

        dataset_input_parameters = {
            'preprocess': {
                'base': dataset_input_parameters,
                'mshi': mshi
            }
        }
            # dataset_input_parameters = {
                # 'sample': {
                    # 'base': dataset_input_parameters,
                # }
            # }

        dataset_input_settings = dataset.dataset_settings_factory(
            dataset_input_parameters)
        interactions_df = dataset.parquet_load(
            dataset_input_settings['interactions_path'])
        # print()
        num_users = len(set(interactions_df.user_id.unique()) | set(interactions_df.user_id.unique()))
        num_items = len(set(interactions_df.item_id.unique()) | set(interactions_df.item_id.unique()))

        total_observations = len(interactions_df)
        print(dataset_name,mshi)
        print('Num users: {} Num items: {} Num user-item pair unique observations: {}'.
              format(num_users, num_items, total_observations))
        print('Train Sparsity: {:.7%}'.format(1 - total_observations /
                                              (num_users * num_items)))
