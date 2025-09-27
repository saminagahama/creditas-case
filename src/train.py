import numpy as np
import pandas as pd
from src import config
from src import data_processing
from src import pipeline as model_pipeline

def run_training():
    raw_df = data_processing.load_data(config.DATA_PATH)
    processed_df = data_processing.preprocess_data(raw_df)
    
    X_train, X_test, y_train, y_test = data_processing.split_data(processed_df)
    
    X_train.drop(columns=config.FEATURES_TO_EXCLUDE, inplace=True, errors='ignore')
    X_test.drop(columns=config.FEATURES_TO_EXCLUDE, inplace=True, errors='ignore')
    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    pipeline = model_pipeline.create_model_pipeline(numeric_features, categorical_features)
    trained_pipeline = model_pipeline.train_model(pipeline, X_train, y_train)
    model_pipeline.evaluate_model(trained_pipeline, X_test, y_test)
    model_pipeline.save_model(trained_pipeline, config.MODEL_PATH)

if __name__ == '__main__':
    run_training()