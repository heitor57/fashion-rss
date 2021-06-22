


RATE = 0.01

RANDOM_SEED = 1

# negative_samples_file = 'data_phase1/data/negative_samples_file.parquet'

EMBEDDING_DIM = 30

dataset_parameters = {
    # 'rate': constants.RATE,
    # 'random_seed': constants.RANDOM_SEED,
    # 'train_path_name': 'data_phase1/train.parquet',
    # 'test_path_name': 'data_phase1/validation.parquet',
    # 'attributes_path_name': 'data_phase1/attributes.parquet',
    'train_path_name': 'data_phase1/data/dummies/train.parquet',
    'test_path_name': 'data_phase1/data/dummies/test.parquet',
    'attributes_path_name': 'data_phase1/data/dummies/attributes.parquet',
    # 'train_path_name': 'data_phase1/data/train.parquet',
    # 'test_path_name': 'data_phase1/data/test.parquet',
    'user_int_ids': 'data_phase1/data/dummies/user_int_ids.pickle',
    'product_int_ids': 'data_phase1/data/dummies/product_int_ids.pickle',
}
