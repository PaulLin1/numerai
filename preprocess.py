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

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'input_path',
    'output_path',
    # 'sagemaker_endpoint'
])

input_path = f's3://{args['input_path']}'
output_path = f's3://{args['output_path']}'
sagemaker_endpoint = args['sagemaker_endpoint']

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

dataframe = datasource.toDF()
print(f"Loaded {dataframe.count()} records")

import sys
sys.exit(0)

def predict_batch(iterator):
    results = []
    for batch in iterator:
        pdf = pd.DataFrame(batch)
        
        # Prepare data for prediction
        # Drop ID columns or columns not needed for prediction if necessary
        # features_pdf = pdf.drop(['id'], axis=1, errors='ignore')
        
        csv_data = pdf.to_csv(index=False, header=False)
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='text/csv',
            Body=csv_data
        )
        
        predictions = json.loads(response['Body'].read().decode())
        
        for i, row in enumerate(batch):
            result = dict(row)
            result['prediction'] = predictions[i] if i < len(predictions) else None
            results.append(result)
    
    return results

batch_size = 100
prediction_rdd = dataframe.rdd.mapPartitions(lambda partition: 
    predict_batch([partition.next() for _ in range(batch_size) if partition.hasNext()]))

predictions_df = spark.createDataFrame(prediction_rdd)

predictions_dyf = DynamicFrame.fromDF(predictions_df, glueContext, "predictions")

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