import pandas as pd
import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
import matplotlib.dates
import sklearn
import sklearn.feature_selection
import re
import joblib
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

# dataset_input_parameters = {'farfetchfinal':{}}
dataset_input_parameters = {'amazon_fashion': {}}
# dataset_input_parameters = {'amazon_cloth': {}}

dataset_output_parameters = {'preprocess': {'base': dataset_input_parameters, 'mshi': 10}}
dataset_output_settings = dataset.dataset_settings_factory(dataset_output_parameters)
interactions_df=dataset.preprocess(dataset_input_parameters=dataset_input_parameters,
                   dataset_output_parameters=dataset_output_parameters)

print('interactions',interactions_df.shape)
print('users',interactions_df.user_id.nunique())
print('items',interactions_df.item_id.nunique())
with open(dataset_output_settings['interactions_path'],'wb') as f:
    interactions_df.to_parquet(f)
