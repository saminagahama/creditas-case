import pickle
import numpy as np
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def create_model_pipeline(numeric_features, categorical_features):
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
    
    lgbm = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgbm)
    ])
    
    return model_pipeline

def train_model(pipeline, X_train, y_train):
    print("Iniciando treinamento do modelo...")
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