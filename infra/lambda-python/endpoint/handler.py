import boto3
import json
import os
# from sagemaker import image_uris

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')
    model_name = event.get("training_job_name")

    s3_uri = f's3://{os.environ.get("S3_URL")}' + '/models/' + model_name + '/output/model.tar.gz'

    model_response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.5-1',
            'ModelDataUrl': s3_uri,
            # 'Environment': {
			# 					'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model',
            #     # 'SAGEMAKER_SUBMIT_DIRECTORY': s3_uri,
            #     'SAGEMAKER_PROGRAM': 'inference.py'
            # }
        },
        ExecutionRoleArn=os.environ.get("SAGEMAKER_ARN")
    )

    # Create endpoint configuration
    endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=f"{model_name}-config",
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large'
            }
        ]
    )
    
    # Create endpoint
    endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=f"{model_name}-endpoint",
        EndpointConfigName=f"{model_name}-config"
    )

    ssm_client = boto3.client("ssm", region_name="your-region")
    ssm_client.put_parameter(
        Name="/numerai/current_endpoint",
        Value=f"{model_name}-endpoint",
        Type="String",
        Overwrite=True
    )
        
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f"Model deployment initiated for endpoint: {model_name}-endpoint",
            'endpoint_name': f"{model_name}-endpoint"
        })
    }
