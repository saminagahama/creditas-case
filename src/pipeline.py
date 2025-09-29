import pickle
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from src.optimized_params import get_lightgbm_params, get_ensemble_params

def create_model_pipeline(numeric_features, categorical_features, model_type="lightgbm"):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    if model_type == "ensemble":
        params = get_ensemble_params()
        pos_weight = params['positive_class_weight']
        class_weights_dict = {0: 1, 1: pos_weight}
        
        xgb_model = XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            n_estimators=params['xgb_n_estimators'],
            max_depth=params['xgb_max_depth'],
            learning_rate=params['xgb_learning_rate'],
            scale_pos_weight=pos_weight
        )
        
        lgbm_model = lgb.LGBMClassifier(
            random_state=42,
            n_estimators=params['lgbm_n_estimators'],
            max_depth=params['lgbm_max_depth'],
            learning_rate=params['lgbm_learning_rate'],
            num_leaves=params['lgbm_num_leaves'],
            scale_pos_weight=pos_weight,
            verbosity=-1
        )
        
        rf_model = RandomForestClassifier(
            random_state=42,
            n_estimators=params['rf_n_estimators'],
            max_depth=params['rf_max_depth'],
            class_weight=class_weights_dict
        )
        
        lr_model = LogisticRegression(
            random_state=42,
            class_weight=class_weights_dict,
            C=params['lr_C'],
            solver='liblinear',
            max_iter=1000
        )
        
        mlp_model = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=params['mlp_hidden_layers'],
            alpha=params['mlp_alpha'],
            learning_rate_init=params['mlp_lr_init'],
            max_iter=300
        )
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('lightgbm', lgbm_model),
                ('random_forest', rf_model),
                ('logistic', lr_model),
                ('mlp', mlp_model)
            ],
            voting='soft'
        )
        
        classifier = ensemble
    else:
        params = get_lightgbm_params()
        classifier = lgb.LGBMClassifier(**params)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return model_pipeline

def train_model(pipeline, X_train, y_train, model_type="lightgbm"):
    if model_type == "ensemble":
        print("Iniciando treinamento do modelo ensemble (5 modelos)...")
        print("Isso pode demorar alguns minutos...")
    else:
        print("Iniciando treinamento do modelo LightGBM...")
    
    pipeline.fit(X_train, y_train)
    print("Treinamento concluído.")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    return {"auc": auc, "average_precision": ap}

def save_model(pipeline, path):
    with open(path, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"\nModelo salvo com sucesso em: {path}")