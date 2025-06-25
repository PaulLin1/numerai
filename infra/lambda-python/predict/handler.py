import boto3
import json
import os
import pandas as pd
import io
import base64

def lambda_handler(event, context):
    sagemaker = boto3.client('sagemaker')
    s3 = boto3.client("s3")
    bucket_name = os.environ.get("S3_NAME")
    
    ssm_client = boto3.client("ssm")
    response = ssm_client.get_parameter(Name="/numerai/current_round")
    current_round = response["Parameter"]["Value"]
    response = ssm_client.get_parameter(Name="/numerai/current_endpoint")
    endpoint_name = response["Parameter"]["Value"]
    
    s3_object = s3.get_object(Bucket=bucket_name, Key=f'data/live/{current_round}.parquet')
    parquet_buffer = io.BytesIO(s3_object["Body"].read())
    df = pd.read_parquet(parquet_buffer)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    response = sagemaker.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=csv_content
    )
    
    prediction_result = response['Body'].read()
    prediction_text = prediction_result.decode('utf-8')
    
    s3.put_object(
        Bucket=bucket_name,
        Key=f'predictions/{current_round}.csv',
        Body=prediction_text,
        ContentType='text/csv'
    )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "message": "Predictions saved successfully",
            "file_location": f"s3://{bucket_name}/predictions/{current_round}.csv"
        })
    }