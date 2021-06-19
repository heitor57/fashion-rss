import dataset 




train_df = dataset.parquet_load(file_name=f'data_phase1/train.parquet')
train_df, test_df = dataset.one_split(train_df,0.8)
dataset.parquet_save(train_df,'data_phase1/data/train_one_split.parquet')
dataset.parquet_save(test_df,'data_phase1/data/test_one_split.parquet')

    # dataset.parquet_load(file_name='data_phase1/attributes.parquet')
