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
dataset_parameters = {
    'train_path_name': 'data_phase1/train.parquet',
    'test_path_name': 'data_phase1/validation.parquet',
    'attributes_path_name': 'data_phase1/attributes.parquet',
}
# rate = 0.1
# dataset.parquet_save(f'data_phase1/train_{rate}.parquet')

# dataset.SampleDF(rate,seed=1).process(dataset.ParquetLoad(file_name='data_phase1/train.parquet').process()))

parameters = {'rate': constants.RATE, 'random_seed': constants.RANDOM_SEED}
parameters_id = utils.parameters_to_str(parameters)

dataset.parquet_save(
    dataset.sample_df(
        dataset.parquet_load(file_name='data_phase1/train.parquet'),
        rate=constants.RATE,
        seed=constants.RANDOM_SEED),
    f'data_phase1/data/train.parquet')
    # f'data_phase1/data/train_{parameters_id}.parquet')
dataset.parquet_save(
    dataset.sample_df(
        dataset.parquet_load(file_name='data_phase1/validation.parquet'),
        rate=constants.RATE,
        seed=constants.RANDOM_SEED),
    # f'data_phase1/data/test_{parameters_id}.parquet')
    f'data_phase1/data/test.parquet')
# dataset.ParquetLoad(file_name='data_phase1/validation.parquet').process()
# dataset.ParquetLoad(file_name='data_phase1/attributes.parquet').process()
