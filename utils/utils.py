from numerai_tools.scoring import numerai_corr, correlation_contribution
from numerapi import NumerAPI
import pandas as pd

def download_datasets():
    napi = NumerAPI()

    for f in napi.list_datasets():
        file_type = f.split('.')[-1]
        if file_type == 'json' or file_type =='parquet':
            napi.download_dataset(f, f'../data/{f}')

def score(validation, DATA_VERSION):
    # Download and join in the meta_model for the validation eras
    # napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
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

    return {
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

def save_model():
    def predict(live_features: pd.DataFrame) -> pd.DataFrame:
        live_predictions = model.predict(live_features[feature_set])
        submission = pd.Series(live_predictions, index=live_features.index)
        return submission.to_frame("prediction")

    import cloudpickle
    p = cloudpickle.dumps(predict)
    with open("hello_numerai.pkl", "wb") as f:
        f.write(p)
