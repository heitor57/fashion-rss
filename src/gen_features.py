from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

base_dir = 'data_phase1/'
attributes_df = pd.read_parquet(base_dir+'attributes.parquet')
train_df = pd.read_parquet(base_dir+'train.parquet')
validation_df = pd.read_parquet(base_dir+'validation.parquet')

user_ids = np.unique(np.hstack((train_df.user_id,validation_df.user_id)))
train_dict = dict()
total = len(train_df["user_id"])
for index, row in tqdm(train_df.iterrows(), position=0, leave=True, total=total):
    if row['user_id'] not in train_dict:
        train_dict[row['user_id']] = []
    train_dict[row['user_id']].append(row)

for user_id in tqdm(train_dict, position=0, leave=True):
    train_dict[user_id] = pd.DataFrame(train_dict[user_id])

user_features = {"uid": [], "items_clicked": np.zeros(len(user_ids)), "observed_items": np.zeros(len(user_ids)),
                 "num_sessions": np.zeros(len(user_ids)), "mean_price": np.zeros(len(user_ids))}

for index, user_id in tqdm(enumerate(user_ids), position=0, leave=True):
    user_features["uid"].append(user_id)
    if user_id in train_dict:
        user_features["items_clicked"][index] = len(train_dict[user_id].loc[(train_dict[user_id].is_click == 1)]["product_id"])
        user_features["observed_items"][index] = len(train_dict[user_id]["user_id"])
        user_features["num_sessions"][index] = len(train_dict[user_id]["session_id"].unique())
        user_features["mean_price"][index] = train_dict[user_id].loc[(train_dict[user_id].is_click == 1)]["product_price"].mean()
   
df_user_features = pd.DataFrame(user_features)
df_user_features.to_csv("df_user_features.csv")