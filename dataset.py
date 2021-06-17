
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import math
import time
import scipy.sparse
import os
import re
import numpy as np
import os
from copy import copy


# class DatasetPreprocessor:
    # def __init__(self, name, dataset_descriptor, preprocessor, *args,
                 # **kwargs):
        # super().__init__(*args, **kwargs)
        # self.name = name
        # self.dataset_descriptor = dataset_descriptor
        # self.preprocessor = preprocessor
        # self.parameters.extend(['dataset_descriptor', 'preprocessor'])

    # def get_id(self, *args, **kwargs):
        # return super().get_id(len(self.parameters), *args, **kwargs)


class Pipeline(Parameterizable):
    def __init__(self, steps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if steps is None:
            self.steps = []
        else:
            self.steps = steps
        self.parameters.extend(['steps'])

    def process(self, data):
        buf = data
        for element in self.steps:
            buf = element.process(buf)
        return buf


# class DatasetDescriptor(Parameterizable):
    # def __init__(self, dataset_dir, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.dataset_dir = dataset_dir
        # self.parameters.extend(['dataset_dir'])


class Dataset:
    def __init__(self,
                 data,
                 num_total_users=None,
                 num_total_items=None,
                 num_users=None,
                 num_items=None,
                 rate_domain=None,
                 uids=None):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids
        self.num_total_users = num_total_users
        self.num_total_items = num_total_items

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[:, 0]))
        self.num_items = len(np.unique(self.data[:, 1]))
        self.rate_domain = set(np.unique(self.data[:, 2]))
        self.uids = np.unique(self.data[:, 0])
        self.mean_rating = np.mean(self.data[:, 2])
        self.min_rating = np.min(self.data[:, 2])
        self.max_rating = np.max(self.data[:, 2])

    def update_num_total_users_items(self):
        self.num_total_users = self.num_users
        self.num_total_items = self.num_items

        # self.consumption_matrix = scipy.sparse.csr_matrix((self.data[:,2],(self..data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))


class DataProcessor(Parameterizable):
    pass


class TRTE(DataProcessor):
    def process(self, dataset_descriptor):
        dataset_dir = dataset_descriptor.dataset_dir
        train_data = np.loadtxt(os.path.join(dataset_dir, 'train.data'),
                                delimiter='::')
        test_data = np.loadtxt(os.path.join(dataset_dir, 'test.data'),
                               delimiter='::')

        dataset = Dataset(np.vstack([train_data, test_data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        train_dataset.update_from_data()
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        test_dataset.update_from_data()
        return train_dataset, test_dataset


class TRTEPopular(DataProcessor):
    def __init__(self, items_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_rate = items_rate
        self.parameters.extend(['items_rate'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        data = np.vstack((test_dataset.data, train_dataset.data))
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items))
        items_popularity = value_functions.MostPopular.get_items_popularity(
            consumption_matrix)
        top_popular_items = np.argsort(
            items_popularity)[::-1][num_items_to_sample]
        test_dataset.data = test_dataset.data[test_dataset.data[:, 1].isin(
            top_popular_items)]
        test_dataset.update_from_data()
        train_dataset.data = train_dataset.data[train_dataset.data[:, 1].isin(
            top_popular_items)]
        train_dataset.update_from_data()

        # train_dataset.data[train_dataset.data[:,1].isin(top_popular_items)]

        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class TRTERandom(DataProcessor):
    def __init__(self, min_ratings, random_seed, probability_keep_item, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.min_ratings = min_ratings
        self.random_seed = random_seed
        self.probability_keep_item = probability_keep_item
        self.parameters.extend(
            ['min_ratings', 'random_seed', 'probability_keep_item'])

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        # ttc = TrainTestConsumption(self.train_size, self.test_consumes,
        # self.crono, self.random_seed)
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class ParquetLoad(DataProcessor):
    def __init__(self, file_name, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name

    def process(self,*args,**kwargs):
        df = pd.read_parquet(self.file_name)
        return df

class FarfetchTrainTestNormalization(DataProcessor):
    def __init__(self, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, train_df, test_df,attributes_df):
        def integer_map(values):
            d = dict()
            for i, v in enumerate(values):
                d[v] = i
            return d
        user_ids = np.unique(np.hstack((train_df.user_id.unique(),test_df.user_id.unique())))
        product_ids = np.unique(np.hstack((train_df.product_id.unique(),test_df.product_id.unique())))
        num_users = len(user_ids)
        num_products = len(product_ids)

        user_int_ids = integer_map(user_ids)
        product_int_ids = integer_map(product_ids)

        train_df.user_id = train_df.user_id.map(lambda x: user_int_ids[x])
        train_df.product_id = train_df.product_id.map(lambda x: product_int_ids[x])

        test_df.user_id = test_df.user_id.map(lambda x: user_int_ids[x])
        test_df.product_id = test_df.product_id.map(
        lambda x: product_int_ids[x])

        train_normalized_df = train_df.groupby(['user_id', 'product_id'
                                               ])['is_click'].sum().reset_index()

        train_normalized_df = train_normalized_df[[
            'user_id', 'product_id', 'is_click'
        ]]

        test_normalized_df = test_df[['user_id', 'product_id']].copy()
        attributes_df.product_id = attributes_df.product_id.map(lambda x: product_int_ids[x])

        return train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids


class NegativeSamples(DataProcessor):
    def __init__(self,  *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def process(self,df,*args,**kwargs):
        df= df.set_index('user_id')
        df['products_sampled'] - 0
        products_id= set(df.product_id.unique())
        for user_id in df.index:
            user_df = df.loc[user_id]
            products_sample_space = products_id - set(user_df.product_id)
            # i = random.randint(0,len(products_sample_space)-1)
            # user_df
            
            products_sampled = random.choice(products_sample_space,size=len(user_df),replace=False)
            df.loc[user_id] = products_sampled
            # sampled_product_id = products_sample_space[i]
        return df
