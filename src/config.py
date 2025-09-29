DATA_PATH = "data/raw/dataset.csv"
MODEL_PATH = "models/modelo_ensemble.pkl" # OU "models/modelo_lightgbm.pkl"
MODEL_TYPE = "ensemble"  # "lightgbm" OU "ensemble"

TARGET = 'sent_to_analysis'
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES_TO_EXCLUDE = [
    'id',
    'pre_approved',
    'zip_code',
    'auto_model',
    'landing_page',
    'landing_page_product'
]

COLS_TO_DROP_IN_CLEANING = ['loan_term', 'marital_status', 'utm_term']

DEBT_COLUMNS = [
    'dishonored_checks', 'expired_debts', 'banking_debts',
    'commercial_debts', 'protests', 'verified_restriction',
    'informed_restriction'
]

# Threshold para classificação (probabilidade mínima para classe positiva)
DEFAULT_THRESHOLD = 0.5