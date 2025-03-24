import boto3
import json
import os
from datetime import datetime
# from sagemaker import image_uris

def lambda_handler(event, context):
    sagemaker = boto3.client('sagemaker')
    s3 = boto3.client("s3")

    role = os.environ.get("SAGEMAKER_ARN")
    bucket = f's3://{os.environ.get("S3_NAME")}'
    input_data = bucket + '/data/live'

    ssm_client = boto3.client("ssm")
    response = ssm_client.get_parameter(Name="/numerai/current_round")
    current_round = response["Parameter"]["Value"]

    response = ssm_client.get_parameter(Name="/numerai/current_endpoint")
    endpoint_name = response["Parameter"]["Value"]

    s3_object = s3.get_object(Bucket=os.environ.get("S3_NAME"), Key=f'data/live/{current_round}.parquet')
    csv_stream = s3_object["Body"]

    response = sagemaker.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=csv_stream
    )

    response = {
        "statusCode": 200,
        "body": json.dumps(f"Training jobstarted!")
    }

    return response
