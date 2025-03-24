"""
Download the live data
Discards the previous live data to save space
"""

import pandas as pd
from numerapi import NumerAPI
from dotenv import load_dotenv
import os


def live_predict(data_version):
    napi = NumerAPI('[your api public id]', '[your api secret key]')

    current_round = napi.get_current_round()

    napi.download_dataset(f'{data_version}/live_{current_round}.parquet')
    # live_data = pd.read_parquet(f'{data_version}/live_{current_round}.parquet')
    # live_features = live_data[[f for f in live_data.columns if 'feature' in f]]

    # live_predictions = model.predict(live_features)
    # submission = pd.Series(live_predictions, index=live_features.index).to_frame('prediction')
    # submission.to_csv(f'prediction_{current_round}.csv')

    # napi.upload_predictions(f'prediction_{current_round}.csv', model_id='your-model-id')

if __name__ == '__main__':
    load_dotenv()
    data_version = os.getenv('DATA_VERSION')
    live_predict(data_version)