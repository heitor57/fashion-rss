import dataset 
dataset_input_name= 'farfetch'
dataset_input_parameters= {}
input_dataset_settings = dataset.dataset_settings_factory(dataset_input_name,dataset_input_parameters)

dataset_output_name= 'split'
dataset_output_parameters = {'base': {dataset_input_name:dataset_input_parameters},'train_size':0.8 }
output_dataset_settings = dataset.dataset_settings_factory(dataset_output_name,dataset_output_parameters)


train_df = dataset.parquet_load(file_name=input_dataset_settings['train_path'])
train_df, test_df = dataset.one_split(train_df,dataset_output_parameters['train_size'])
dataset.parquet_save(train_df,output_dataset_settings['train_path'])
dataset.parquet_save(test_df,output_dataset_settings['validation_path'])

    # dataset.parquet_load(file_name='data_phase1/attributes.parquet')
