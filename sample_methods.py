import constants
def sample_fixed_size(df, num_samples):

    return df.sample(num_samples, random_state = constants.RANDOM_SEED)

