from yaml.loader import Loader
import torch
import yaml
from itertools import chain
import numpy as np
import pyximport; pyximport.install()
import metrics
from tqdm import tqdm
import loss_functions
import json


from pathlib import Path
import os.path

def create_path_to_file(file_name,until=-1):
    Path('/'.join(file_name.split('/')[:until])).mkdir(parents=True,
                                                    exist_ok=True)
def file_exists(fname):
    return os.path.isfile(fname) 

def parameters_to_str(d):
    return json.dumps(d,separators=(',', ':'))


def BPR_train_original(dataset, neural_network, loss_class, epoch, neg_k=1):
    bpr: loss_functions.BPRLoss = loss_class

    # with timer(name="Sample"):
    # S = utils.UniformSample_original(dataset)
    S = dataset
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        # if world.tensorboard:
        # w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    # time_info = timer.dict()
    # timer.zero()
    return f"loss{aver_loss:.3f}"


def eval_mrr(results_df,test_df):
    test_dict = test_df.set_index(['user_id','item_id'])['target'].to_dict()
    mrr =0
    c =0
    for name, group in tqdm(results_df.groupby('user_id')):
        
        user_id = group['user_id'].iloc[0]
        # print(all_product_is_click)
        # is_click=all_product_is_click[query_id]
        # print(is_click)
        # print(group['rank'])
        # print(group.product_id)
        # print(is_click)
        mrr+= metrics.cython_rr(group.user_id.values,group.item_id.values,group['rank'].values,test_dict)
        c+=1

    # print(mrr)
    # print(c)
    # print('mrr',mrr/c)
    return mrr/c

        
def eval_ndcg(results_df,test_df):
    test_df = test_df.loc[test_df.target>0]
    test_dict = test_df.set_index(['user_id'])['item_id'].to_dict()
    values_acc =0
    count =0
    for name, group in tqdm(results_df.groupby('user_id')):
        
        user_id = group['user_id'].iloc[0]
        # print(all_product_is_click)
        # is_click=all_product_is_click[query_id]
        # print(is_click)
        # print(group['rank'])
        # print(group.product_id)
        # print(is_click)
        # values_acc+= metrics.ndcgk(set([test_dict[user_id]]),group.item_id.values,len(group))
        values_acc+= metrics.ndcgk(set([test_dict[user_id]]),group.item_id.values,len(group))
        count+=1

    return values_acc/count

def eval_hits(results_df,test_df):
    test_df = test_df.loc[test_df.target>0]
    test_dict = test_df.set_index(['user_id'])['item_id'].to_dict()
    values_acc =0
    count =0
    for name, group in tqdm(results_df.groupby('user_id')):
        
        user_id = group['user_id'].iloc[0]
        # print(all_product_is_click)
        # is_click=all_product_is_click[query_id]
        # print(is_click)
        # print(group['rank'])
        # print(group.product_id)
        # print(is_click)
        # values_acc+= metrics.ndcgk(set([test_dict[user_id]]),group.item_id.values,len(group))
        values_acc+= test_dict[user_id] in group.item_id.values
        count+=1

    return values_acc/count

def generate_negative_samples(test_df,users,num_negatives,interactions_matrix,num_items):
    negatives = []
    for i in range(len(test_df)):
        user = users[i]
        for _ in range(num_negatives):
            while True:
                item = np.random.randint(num_items)
                if (user, item) not in interactions_matrix:
                    negatives.append([user, item, 0,test_df.iloc[i].day,test_df.iloc[i].week])
    # interactions_matrix[user, item] = 0
                    break
    return negatives

def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]



def load_best_parameters():
    loader = yaml.SafeLoader
    path = "./data/settings.yaml"
    if file_exists(path):
        d = yaml.load(open(path),Loader=loader)
    else:
        d = {}
    return d

def save_best_parameters(d):
    open('./data/settings.yaml','w').write(yaml.dump(d))
    return d

