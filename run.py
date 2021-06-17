import pandas as pd
import neural_networks
import recommenders
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import dataset

train_normalized_df, test_normalized_df, attributes_df, user_int_ids, product_int_ids = dataset.FarfetchTrainTestNormalization().process(
dataset.ParquetLoad(file_name='data_phase1/train.parquet').process(),
dataset.ParquetLoad(file_name='data_phase1/validation.parquet').process(),
dataset.ParquetLoad(file_name='data_phase1/attributes.parquet').process()
)


print(dataset.NegativeSamples().process(train_normalized_df))
# num_users = len(user_int_ids)
# num_items = len(product_int_ids)
# embedding_dim = 30
# recommenders.NNRecommender(neural_networks.BilinearNet(num_users,num_items,embedding_dim,sparse=True))



# plt.style.reload_library()
# plt.style.use(['science', 'notebook'])
# plt.rcParams['figure.figsize'] = [12 / 2.5, 8 / 2.5]
# plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
# base_dir = 'data_phase1/'
# attributes_df = pd.read_parquet(base_dir + 'attributes.parquet')
# train_df = pd.read_parquet(base_dir + 'train.parquet')
# validation_df = pd.read_parquet(base_dir + 'validation.parquet')

# # validation_normalized_df.loc[:, 'tmp_rating'] = 1
# embedding_dim = 20

# train = train_input_normalized_df
# test = validation_normalized_df
# recommender = recommenders.BilinearNet(num_users,num_products, embedding_dim)

# data = []
# for uid, items in prediction_items.items():
    # indexes = np.argsort(prediction_scores[uid])[::-1]
    # items = [items[i] for i in indexes]
    # for i, item in enumerate(items):
        # x = [uid, item, i + 1]
        # data.append(x)

# df = pd.DataFrame(data)
# df.columns = ['user_id', 'product_id', 'rank']


# # df['user_id'].unique()
# def _map(x):
    # if len(x) > 1:
        # raise SystemError
    # else:
        # return x[0]


# user_id_to_query_id = validation_df.groupby('user_id')['query_id'].unique().map(
    # _map).to_dict()

# tmp = {v: k for k, v in product_int_ids.items()}
# df['user_id'] = df['user_id'].map(lambda x: user_id_to_query_id[x])
# df['product_id'] = df['product_id'].map(lambda x: tmp[x])
# df.columns = ['query_id', 'product_id', 'rank']
# df.to_csv('data_phase1/output.csv', index=False)




