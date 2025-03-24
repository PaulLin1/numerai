"""
Train and scores a new model
Use CLI arguments for other stuff
"""

import json
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from numerai_tools.scoring import numerai_corr, correlation_contribution
from numerapi import NumerAPI
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv
import joblib
import argparse

load_dotenv()
DATA_VERSION = os.getenv('DATA_VERSION')

def train():
    # feature_set is not needed because train_small.parquet only contains small feature_set
    # feature_set = json.load(open(f"../data/{DATA_VERSION}/features_small.json"))

    train = pd.read_parquet(
        f"../data/{DATA_VERSION}/train_small.parquet",
        # columns=["era", "target"] + feature_set
    )

    features_df = train.copy()
    features_df = features_df.drop(['target', 'era'], axis=1)

    model = lgb.LGBMRegressor()

    param_grid = {
        'learning_rate': [0.01],
    }

    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'n_estimators': [100, 200, 500],
    #     'max_depth': [3, 5, 7],
    #     'num_leaves': [31, 50, 70],
    #     'colsample_bytree': [0.6, 0.8, 1.0],
    # }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='accuracy',
        cv=3,
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(features_df, train['target'])

    return grid_search.best_estimator_

def score(model):
    # Download and join in the meta_model for the validation eras
    # napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)

    validation = pd.read_parquet(
        f"../data/{DATA_VERSION}/validation_small.parquet",
    )

    validation["prediction"] = model.predict(validation[[col for col in validation.columns if col not in ['target', 'era', 'prediction']]])
    validation[["era", "prediction", "target"]]

        
    validation["meta_model"] = pd.read_parquet(
        f"../data/{DATA_VERSION}/meta_model.parquet"
    )["numerai_meta_model"]

    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )

    per_era_mmc = validation.dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
    )

    # Compute performance metrics
    corr_mean = per_era_corr.mean()
    corr_std = per_era_corr.std(ddof=0)
    corr_sharpe = corr_mean / corr_std
    corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

    mmc_mean = per_era_mmc.mean()
    mmc_std = per_era_mmc.std(ddof=0)
    mmc_sharpe = mmc_mean / mmc_std
    mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

    metrics =  {
        "corr": {
            "mean": corr_mean['prediction'],
            "std": corr_std['prediction'],
            "sharpe": corr_sharpe['prediction'],
            "max_drawdown": corr_max_drawdown['prediction']
            },
        "mmc": {
            "mean": mmc_mean['prediction'],
            "std": mmc_std['prediction'],
            "sharpe": mmc_sharpe['prediction'],
            "max_drawdown": mmc_max_drawdown['prediction']
            }
    }

    return metrics

if __name__ == '__main__':
    print('local training job started')

    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-sm", type=int, default=0, help="Pulls current SageMaker model to score (optional)")
    parser.add_argument("--upload-model", type=int, default=0, help="Upload model to SageMaker (optional)")
    parser.add_argument("--create-endpoint", type=int, default=0, help="Create SageMaker endpoint (optional)")

    args = parser.parse_args()

    # Makes a new folder for new model
    # Date and then suffix if model was already trained on that date
    CURRENT_TIME = datetime.now().strftime("%m_%d_%y")
    folder_path = f"../models/{CURRENT_TIME}"
    original_folder_path = folder_path
    suffix = 1
    while os.path.exists(folder_path):
        folder_path = f'{original_folder_path}_{suffix}'
        suffix += 1
    os.makedirs(folder_path, exist_ok=True)

    # Train and dump model
    model = train()
    model_path = f"{folder_path}/model.lgb"
    joblib.dump(model, model_path)
    print('model trained and stored')
 
    # If pull-old = 1, pull current cloud model and score it
    if args.pull_sm == 1:
        score(old_model)

    # Score Model
    # score(model)

    bucket_name = os.getenv('S3_BUCKET_NAME')
    s3_model_path = "models/lightgbm/model.lgb"
    s3_uri = f"s3://{bucket_name}/{s3_model_path}"

    # If upload-model = 1, upload the model to S3
    if args.upload_model == 1:
        import boto3

        s3 = boto3.client("s3")

        # Upload model file to S3
        s3.upload_file(model_path, bucket_name, s3_model_path)

        print("Model uploaded to:", s3_uri)
    
    s3_uri = f"s3://{bucket_name}/models/lightgbm/model.tar.gz"

    # If upload-model = 1, create a SageMaker endpoint
    # NGL, should only do this once and forget about it
    if args.create_endpoint == 1:
        import sagemaker
        from sagemaker.sklearn.model import SKLearnModel

        model = SKLearnModel(
            model_data=s3_uri,
            role=os.getenv('SAGEMAKER_ARN'),
            entry_point="inference.py",
            framework_version="0.23-1",
            dependencies=["requirements.txt"]
        )

        # Deploy model
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large"
        )
