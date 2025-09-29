import os
import sys
from src import config
from src.train import run_training

def train_all_models():
    original_model_type = config.MODEL_TYPE
    original_model_path = config.MODEL_PATH
    
    try:
        # treinar LightGBM
        print("="*60)
        print("TREINANDO MODELO LIGHTGBM")
        print("="*60)
        config.MODEL_TYPE = "lightgbm"
        config.MODEL_PATH = "models/modelo_lightgbm.pkl"
        run_training()
        
        # treinar Ensemble
        print("\n" + "="*60)
        print("TREINANDO MODELO ENSEMBLE")
        print("="*60)
        config.MODEL_TYPE = "ensemble"
        config.MODEL_PATH = "models/modelo_ensemble.pkl"
        run_training()
        
        print("\n" + "="*60)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*60)
        print("Modelos salvos:")
        print("  - models/modelo_lightgbm.pkl")
        print("  - models/modelo_ensemble.pkl")
        print("\nPara alternar entre modelos, edite MODEL_PATH em src/config.py")
        
    finally:
        config.MODEL_TYPE = original_model_type
        config.MODEL_PATH = original_model_path

if __name__ == '__main__':
    train_all_models()