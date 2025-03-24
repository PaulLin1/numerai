import boto3
import json
import os
import os
import io
import requests

def lambda_handler(event, context):
    url = "https://api-tournament.numer.ai"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    round_data = {
        "query": "{rounds(limit:1) {number}}"
    }
    round_response = requests.post(url, json=round_data, headers=headers).json()
    current_round = round_response['data']['rounds'][0]['number']

    ssm_client = boto3.client("ssm", region_name="your-region")
    ssm_client.put_parameter(
        Name="/numerai/current_round",
        Value=current_round,
        Type="String",
        Overwrite=True
    )

    live_data = {
        "query": f"{{ dataset(filename: \"v5.0/live.parquet\", round: {current_round}) }}"
    }
    live_response = requests.post(url, json=live_data, headers=headers).json()
    live_url = live_response['data']['dataset']

    response = requests.get(live_url)

    if response.status_code == 200:
        s3_client = boto3.client('s3')
        s3_bucket = os.environ.get("S3_NAME")
        s3_key = f'data/live/{current_round}.parquet'

        # Create a buffer to store the file content
        buffer = io.BytesIO()
        buffer.write(response.content)
        buffer.seek(0)  # Reset the buffer pointer to the beginning

        # Upload the content to S3
        s3_client.put_object(Body=buffer, Bucket=s3_bucket, Key=s3_key)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Data downloaded and uploaded to S3 for round {current_round}"
            })
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Failed to download the parquet file from the live URL'
            })
        }