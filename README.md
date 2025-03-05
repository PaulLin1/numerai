# Numerai Pipeline

My end-to-end Numerai prediction pipeline. Focus is staying accurate and keeping cloud costs down while still using SageMaker. A lot of downsampling and some offline training. Documentation for myself.

## Goal
End to end
    Basically everything is automated and done in the cloud. Big exception is downloading the main training data. That is done once because it doesn't get updated too much. Retraining will look at new features and integrate the new live data.

Accuracy
    The model should be accurate and do well on the Numerai metrics. However, I know I cannot get competitive results because my lack of compute compared to others. For now, I will focus on the infra and use a simple LGBM model with grid search.

## Folders
### local_data
Instead of downloading all the data directly to my S3 bucket, I download it locally, downsample it, and then upload it to my S3 bucket. This saves a lot of cost. This is one of the few parts that is not automated. However, this is not a big issue because new versions of the training data do not come out that often.

### notebooks
Just offline experiments. Not sagemaker notebooks.

### utils
Nothing is used in the pipeline. If they do end up getting used, they will be converted to a lambda function. Some are notebooks for data visualization. They are not in the notebooks folder because they are more used and experiments.

### models
Local experiment models. These are not used in the pipeline.