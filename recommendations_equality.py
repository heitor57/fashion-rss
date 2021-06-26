import dataset
from tqdm import tqdm
import argparse
import scipy.stats
import pandas as pd

argparser = argparse.ArgumentParser()
# argparser.add_argument('-b', type=str)
argparser.add_argument('-m1', type=str)
argparser.add_argument('-m2', type=str)
args = argparser.parse_args()




# results_df.to_csv(f'data_phase1/data/{method}_output.csv'
results_df1 =  pd.read_csv(f'data_phase1/data/{args.m1}_output.csv')
results_df1 = results_df1.sort_values(['query_id','product_id']).set_index('query_id')
results_df2 =  pd.read_csv(f'data_phase1/data/{args.m2}_output.csv')
results_df2 = results_df2.sort_values(['query_id','product_id']).set_index('query_id')

kendall = 0
c=0
values = results_df2.index
for i in tqdm(values):
    rank1 = results_df1.loc[i]['rank'].values
    rank2 = results_df2.loc[i]['rank'].values
    kendall += scipy.stats.kendalltau(rank1,rank2)[0]
    c +=1
    if c % 100000 == 0:
        print('kendall',kendall/c)
print('kendall',kendall/c)
