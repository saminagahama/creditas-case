import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    TARGET, RANDOM_STATE, TEST_SIZE, COLS_TO_DROP_IN_CLEANING, DEBT_COLUMNS
)

def load_data(path):
    return pd.read_csv(path)

def _clean_data(df):
    df_cleaned = df.copy()
    df_cleaned.drop(columns=COLS_TO_DROP_IN_CLEANING, inplace=True, errors='ignore')

    nan_imputation_numeric = {
        'collateral_debt': 0, 'verified_restriction': 0, 'informed_restriction': 0
    }
    df_cleaned.fillna(nan_imputation_numeric, inplace=True)

    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col].fillna('Unknown', inplace=True)
    
    initial_rows = len(df_cleaned)
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.drop_duplicates(subset=['id'], keep='first', inplace=True)
    
    return df_cleaned

def feature_engineering(df):
    df_eng = df.copy()
    
    df_eng['total_debts'] = df_eng[DEBT_COLUMNS].sum(axis=1)
    df_eng['has_any_debt'] = (df_eng['total_debts'] > 0).astype(int)
    
    df_eng['loan_to_income_ratio'] = df_eng['loan_amount'] / (df_eng['monthly_income'] + 1e-6)
    df_eng['loan_to_value_ratio'] = df_eng['loan_amount'] / (df_eng['collateral_value'] + 1e-6)
    df_eng['income_to_value_ratio'] = df_eng['monthly_income'] / (df_eng['collateral_value'] + 1e-6)
    
    df_eng['car_age'] = df_eng['auto_year'].max() - df_eng['auto_year']
    
    return df_eng

def preprocess_data(df):
    df_processed = df.copy()
    if 'pre_approved' in df_processed.columns:
        df_processed = df_processed[df_processed['pre_approved'] == True].copy()
    df_processed = _clean_data(df_processed)
    df_processed = feature_engineering(df_processed)
    
    return df_processed

def split_data(df):
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test