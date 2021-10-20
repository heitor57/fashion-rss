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

# argparser = argparse.ArgumentParser()
# argparser.add_argument('-sb', type=str)
# argparser.add_argument('-tb', type=str)
# args = argparser.parse_args()
# # train_size=0.9
# args.sb= json.loads(args.sb)
# args.tb= json.loads(args.tb)
# sb_settings = dataset.dataset_settings_factory(args.sb)
# tb_settings = dataset.dataset_settings_factory(args.tb)

# dataset_1_parameters= {'farfetch':{}}
# dataset_output_name= 'split'
dataset_input_parameters = {'farfetch':{}}
# dataset_input_parameters = dataset_output_parameters
dataset_input_settings = dataset.dataset_settings_factory(dataset_input_parameters)

dataset_output_parameters = {'split':{'base': dataset_input_parameters,'train_size':0.8}}
dataset_output_settings = dataset.dataset_settings_factory(dataset_output_parameters)


train_df, validation_df = dataset.one_split(
    dataset.parquet_load(file_name=dataset_input_settings['train_path']), list(dataset_output_parameters.values())[0]['train_size'])

# out_dataset_settings = dataset.dataset_settings_factory('split',{'train_size': train_size, 'base_name': args.sb})

# print(train_df,validation_df)
dataset.parquet_save(train_df, dataset_output_settings['train_path'])
dataset.parquet_save(validation_df, dataset_output_settings['validation_path'])
