import numpy as np
import pandas as pd
import sys
from src import config
from src import data_processing
from src import pipeline as model_pipeline

def run_training(model_type=None, model_path=None):
    if model_type is None:
        model_type = config.MODEL_TYPE
    if model_path is None:
        model_path = config.MODEL_PATH
    
    raw_df = data_processing.load_data(config.DATA_PATH)
    processed_df = data_processing.preprocess_data(raw_df)
    
    X_train, X_test, y_train, y_test = data_processing.split_data(processed_df)
    
    X_train.drop(columns=config.FEATURES_TO_EXCLUDE, inplace=True, errors='ignore')
    X_test.drop(columns=config.FEATURES_TO_EXCLUDE, inplace=True, errors='ignore')
    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    pipeline = model_pipeline.create_model_pipeline(numeric_features, categorical_features, model_type)
    trained_pipeline = model_pipeline.train_model(pipeline, X_train, y_train, model_type)
    model_pipeline.evaluate_model(trained_pipeline, X_test, y_test)
    model_pipeline.save_model(trained_pipeline, model_path)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type == "lightgbm":
            run_training("lightgbm", "models/modelo_lightgbm.pkl")
        elif model_type == "ensemble":
            run_training("ensemble", "models/modelo_ensemble.pkl")
        else:
            print("Uso: python -m src.train [lightgbm|ensemble]")
            print("Ou apenas: python -m src.train (usa configuração do config.py)")
    else:
        run_training()