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
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('-sb', type=str)
args = argparser.parse_args()
train_size=0.9
dataset_settings = dataset.dataset_settings_factory(args.sb)
train_df, validation_df = dataset.one_split(
    dataset.parquet_load(file_name=dataset_settings['train_path_name']), train_size)

out_dataset_settings = dataset.dataset_settings_factory('split',{'train_size': train_size, 'base_name': args.sb})

dataset.parquet_save(train_df, out_dataset_settings['train_path_name'])
dataset.parquet_save(validation_df, out_dataset_settings['validation_path_name'])
