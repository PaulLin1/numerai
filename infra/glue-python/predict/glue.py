import sys
import boto3
import json
import pandas as pd
from pyspark.sql import SparkSession
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

# Get job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'SAGEMAKER_ENDPOINT', 'INPUT_S3', 'OUTPUT_S3'])

# Initialize Spark and Glue
spark = SparkSession.builder.appName("GlueXGBoostJob").getOrCreate()
glueContext = GlueContext(spark)
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# SageMaker client
sagemaker_client = boto3.client('sagemaker-runtime')

# Read input data from S3
input_df = spark.read.csv(args['INPUT_S3'], header=True, inferSchema=True)

# Convert to Pandas for easier batch processing
pandas_df = input_df.toPandas()

# Function to invoke SageMaker endpoint
def predict(row):
    payload = json.dumps(row.tolist())  # Convert row to list then JSON
    response = sagemaker_client.invoke_endpoint(
        EndpointName=args['SAGEMAKER_ENDPOINT'],
        ContentType='application/json',
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    return result['predictions'][0]  # Adjust if response structure differs

# Apply prediction function
pandas_df['prediction'] = pandas_df.apply(predict, axis=1)

# Convert back to Spark DataFrame and save to S3
output_df = spark.createDataFrame(pandas_df)
output_df.write.mode('overwrite').csv(args['OUTPUT_S3'], header=True)

# Commit job
job.commit()
