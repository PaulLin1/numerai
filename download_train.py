'''
Downloads the training and validation data and the meta model
This should only be run when a new dataset comes out
Change DATA_VERSION to the appropriate data_version

This does not download the live data as it comes out more frequently
The live data will be downloaded directly into the S3 bucket
'''

import json
import pandas as pd
from numerapi import NumerAPI

# Where the data is stored locally
DATA_PATH = 'local_data'

DATA_VERSION = 'v5.0'
napi = NumerAPI()

# Download features
napi.download_dataset(f'{DATA_VERSION}/features.json', f'{DATA_PATH}/{DATA_VERSION}/features.json')
feature_metadata = json.load(open(f'{DATA_PATH}/{DATA_VERSION}/features.json'))
feature_set = feature_metadata['feature_sets']['small'] # Smallest feature set

# Save small feature_set somewhere. I will never use bigger feature set.
with open(f'{DATA_PATH}/{DATA_VERSION}/features_small.json', 'w') as f:
    json.dump(feature_set, f, indent=4)

# import sys
# sys.exit()

# Download training data
napi.download_dataset(f'{DATA_VERSION}/train.parquet', f'{DATA_PATH}/{DATA_VERSION}/train.parquet')
train = pd.read_parquet(
    f'{DATA_PATH}/{DATA_VERSION}/train.parquet',
    columns=['era', 'target'] + feature_set
)
# Downsample to every 4th era to reduce memory usage and speedup model training
train = train[train['era'].isin(train['era'].unique()[::4])]

# Save downsampled and small feature set only train parquet file
train.to_parquet(f'{DATA_PATH}/{DATA_VERSION}/train_small.parquet')

# Download validation data
napi.download_dataset(f'{DATA_VERSION}/validation.parquet', f'{DATA_PATH}/{DATA_VERSION}/validation.parquet')
validation = pd.read_parquet(
    f'{DATA_PATH}/{DATA_VERSION}/validation.parquet',
    columns=['era', 'data_type', 'target'] + feature_set
)
validation = validation[validation['data_type'] == 'validation']
del validation['data_type']

# Downsample to every 4th era to reduce memory usage and speedup evaluation
validation = validation[validation['era'].isin(validation['era'].unique()[::4])]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to 'embargo' the first 4 eras following our last train era to avoid 'data leakage'
last_train_era = int(train['era'].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation['era'].isin(eras_to_embargo)]

# Save downsampled and small feature set only validation parquet file
validation.to_parquet(f'{DATA_PATH}/{DATA_VERSION}/validation_small.parquet')

# Download current meta_model
napi.download_dataset(f'{DATA_VERSION}/meta_model.parquet',f'{DATA_PATH}/{DATA_VERSION}/meta_model.parquet')