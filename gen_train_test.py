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
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('-sb', type=str)
argparser.add_argument('-tb', type=str)
args = argparser.parse_args()
# train_size=0.9
args.sb= json.loads(args.sb)
args.tb= json.loads(args.tb)
sb_settings = dataset.dataset_settings_factory(args.sb)
tb_settings = dataset.dataset_settings_factory(args.tb)
train_df, validation_df = dataset.one_split(
    dataset.parquet_load(file_name=sb_settings['train_path']), list(args.tb.values())[0]['train_size'])

# out_dataset_settings = dataset.dataset_settings_factory('split',{'train_size': train_size, 'base_name': args.sb})

dataset.parquet_save(train_df, tb_settings['train_path'])
dataset.parquet_save(validation_df, tb_settings['validation_path'])
