import boto3
import json
import os
from datetime import datetime
# from sagemaker import image_uris

def lambda_handler(event, context):
    sagemaker = boto3.client('sagemaker')

    training_job_name = f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    role = os.environ.get("SAGEMAKER_ARN")
    bucket = f's3://{os.environ.get("S3_NAME")}'
    input_data = bucket + '/data/v5.0'
    output_data = bucket + '/models'

    # region = boto3.session.Session().region_name
    # xgboost_image = image_uris.retrieve('xgboost', region, '1.5-1')
    # print(xgboost_image)
    # import sys
    # sys.exit(0)

    algorithm_spec = {
        'TrainingImage': '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.5-1',
        'TrainingInputMode': 'File'
    }

    input_config = [{
        'ChannelName': 'train',
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': input_data,
                'S3DataDistributionType': 'FullyReplicated'
            }
        },
        'ContentType': 'text/csv'
    }]

    output_config = {
        'S3OutputPath': output_data
    }

    resource_config = {
        'InstanceType': 'ml.m5.large',  # Instance type for training
        'InstanceCount': 1,             # Number of instances
        'VolumeSizeInGB': 50            # Size of EBS volume in GB
    }

    stopping_condition = {
        'MaxRuntimeInSeconds': 3600  # Max time in seconds, e.g., 1 hour
    }

    hyperparameters = {
        'objective': 'reg:squarederror',  # Regression objective
        'num_round': '20',                  # Number of boosting rounds (as int)
        'max_depth': '6',                    # Maximum depth of a tree
        'eta': '0.3',                        # Step size shrinkage
        'eval_metric': 'rmse',             # Root mean squared error for regression
    }

    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification=algorithm_spec,
        HyperParameters=hyperparameters,
        InputDataConfig=input_config,
        OutputDataConfig=output_config,
        ResourceConfig=resource_config,
        StoppingCondition=stopping_condition,
        RoleArn=role
    )

    response = {
        "statusCode": 200,
        "training_job_name": training_job_name,
        "body": json.dumps(f"Training job '{training_job_name}' started!")
    }

    return response
