import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
import boto3
import pandas as pd
from io import StringIO
import json

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'input_path',
    'output_path',
])

input_path = f"s3://{args['input_path']}validation.parquet"
output_path = f"s3://{args['output_path']}"

ssm_client = boto3.client("ssm")
response = ssm_client.get_parameter(Name="/numerai/current_endpoint")
sagemaker_endpoint = response["Parameter"]["Value"]

sagemaker_runtime = boto3.client('sagemaker-runtime')

print(f"Reading data from {input_path}")
datasource = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [input_path]},
    format="parquet",
    format_options={
        "withHeader": True,
        "separator": ","
    }
)

# Convert to DataFrame for easier manipulation
dataframe = datasource.toDF()
print(f"Loaded {dataframe.count()} records")

# import sys
# sys.exit(0)

# Define a function to batch predictions
def predict_batch(iterator):
    results = []
    for batch in iterator:
        # Convert batch to pandas DataFrame
        pdf = pd.DataFrame(batch)
        
        # Prepare data for prediction
        # Drop ID columns or columns not needed for prediction if necessary
        # features_pdf = pdf.drop(['id'], axis=1, errors='ignore')
        
        # Convert to CSV format for SageMaker
        csv_data = pdf.to_csv(index=False, header=False)
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='text/csv',
            Body=csv_data
        )
        
        # Parse response
        predictions = json.loads(response['Body'].read().decode())
        
        # Combine original data with predictions
        for i, row in enumerate(batch):
            result = dict(row)
            result['prediction'] = predictions[i] if i < len(predictions) else None
            results.append(result)
    
    return results

# Process data in batches to avoid memory issues
batch_size = 100  # Adjust based on your data size and memory constraints
prediction_rdd = dataframe.rdd.mapPartitions(lambda partition: 
    predict_batch([partition.next() for _ in range(batch_size) if partition.hasNext()]))

# Convert back to DataFrame
predictions_df = spark.createDataFrame(prediction_rdd)

# Convert to DynamicFrame
predictions_dyf = DynamicFrame.fromDF(predictions_df, glueContext, "predictions")

# Write results back to S3
print(f"Writing predictions to {output_path}")
glueContext.write_dynamic_frame.from_options(
    frame=predictions_dyf,
    connection_type="s3",
    connection_options={"path": output_path},
    format="csv",
    format_options={
        "separator": ",",
        "quoteChar": '"'
    }
)