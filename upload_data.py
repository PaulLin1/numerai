"""
Upload train data to S3 bucket
"""

import sagemaker
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

data_version = os.getenv('DATA_VERSION')
bucket = os.getenv('S3_BUCKET_URL')

sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
train_path = sess.upload_data(
    path=f'data/{data_version}/train.csv', bucket=bucket, key_prefix=f'data/{data_version}'
)
# validation_path = sess.upload_data(
#     path=f'data/{data_version}/validation_small.csv', bucket=bucket, key_prefix=f'data/{data_version}'
# )