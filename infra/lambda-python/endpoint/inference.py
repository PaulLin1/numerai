import os
import xgboost as xgb
import json
import numpy as np

# Define model loading function
def model_fn(model_dir):
    """Load the trained XGBoost model from the directory"""
    model_path = os.path.join(model_dir, "xgboost-model")  # SageMaker extracts your model to this path
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

# Add input and prediction functions
def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
        
def predict_fn(input_data, model):
    """Make prediction with the model"""
    return model.predict(xgb.DMatrix(input_data))