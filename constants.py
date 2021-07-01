RATE = 0.01

RANDOM_SEED = 1

# negative_samples_file = 'data_phase1/data/negative_samples_file.parquet'

EMBEDDING_DIM = 32

source_dataset_settings = {
    # 'rate': constants.RATE,
    # 'random_seed': constants.RANDOM_SEED,
    # 'train_path': 'data_phase1/train.parquet',
    # 'validation_path': 'data_phase1/validation.parquet',
    # 'attributes_path': 'data_phase1/attributes.parquet',
    'train_path': 'data_phase1/data/dummies/train.parquet',
    'validation_path': 'data_phase1/data/dummies/validation.parquet',
    'attributes_path': 'data_phase1/data/dummies/attributes.parquet',
    # 'train_path': 'data_phase1/data/train.parquet',
    # 'validation_path': 'data_phase1/data/validation.parquet',
    'user_int_ids': 'data_phase1/data/dummies/user_int_ids.pickle',
    'product_int_ids': 'data_phase1/data/dummies/product_int_ids.pickle',
    'query_int_ids': 'data_phase1/data/dummies/query_int_ids.pickle',
}

# datasets_settings = {
    # 'base': {
        # 'train': 'data_phase1/train.parquet',
        # 'validation': 'data_phase1/validation.parquet',
        # 'attributes': 'data_phase1/attributes.parquet',
    # },
    # 'split1': {
        # 'train': 'data_phase1/data/split1_train.parquet',
        # 'validation': 'data_phase1/data/split1_validation.parquet',
        # 'attributes': 'data_phase1/attributes.parquet',
    # },
# }


