import pandas as pd
import utils
import constants
import neural_networks
import recommenders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import dataset

settings = {
    'train': 'data_phase1/train.parquet',
    'validation': 'data_phase1/validation.parquet',
    'attributes': 'data_phase1/attributes.parquet',
}
# rate = 0.1
# dataset.parquet_save(f'data_phase1/train_{rate}.parquet')

# parameters = {'rate': constants.RATE, 'random_seed': constants.RA9DOM_SEED}
# parameters_id = utils.parameters_to_str(parameters)
train_df, validation_df = dataset.one_split(
    dataset.parquet_load(file_name='data_phase1/train.parquet'), 0.9)

dataset.parquet_save(train_df, f'data_phase1/data/train.parquet')
dataset.parquet_save(validation_df, f'data_phase1/data/validation.parquet')
