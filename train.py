import json
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from numerai_tools.scoring import numerai_corr, correlation_contribution
from numerapi import NumerAPI
import pickle
from datetime import datetime
import os

def train():
    # feature_set is not needed because train_small.parquet only contains small feature_set
    # feature_set = json.load(open(f"{DATA_VERSION}/features_small.json"))

    train = pd.read_parquet(
        f"{DATA_VERSION}/train_small.parquet",
        # columns=["era", "target"] + feature_set
    )

    features_df = train.copy()
    features_df = features_df.drop(['target', 'era'], axis=1)

    model = lgb.LGBMRegressor()

    # param_grid = {
    #     'learning_rate': [0.01],
    # }

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 70],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(features_df, train['target'])

    # Save the best model
    with open(f'{folder_path}/model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search.best_estimator_

def score(model):
    # Download and join in the meta_model for the validation eras
    # napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)

    validation = pd.read_parquet(
        f"{DATA_VERSION}/validation_small.parquet",
    )

    validation["prediction"] = model.predict(validation[[col for col in validation.columns if col not in ['target', 'era', 'prediction']]])
    validation[["era", "prediction", "target"]]

        
    validation["meta_model"] = pd.read_parquet(
        f"{DATA_VERSION}/meta_model.parquet"
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

    with open(f'{folder_path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    # Local data is stored under local_data folder
    # DATA_VERSION should just be the v
    #  whatever
    DATA_VERSION = 'local_data/v5.0'

    # Makes a new folder for each model
    # Date and then suffix if model was already trained on that date
    CURRENT_TIME = datetime.now().strftime("%m_%d_%y")
    folder_path = f'models/{CURRENT_TIME}'

    suffix = 1
    original_folder_path = folder_path
    while os.path.exists(folder_path):
        folder_path = f'{original_folder_path}_{suffix}'
        suffix += 1

    os.makedirs(folder_path, exist_ok=True)

    model = train()
    print('model trained')

    score(model)
