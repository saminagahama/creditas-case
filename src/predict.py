import pandas as pd
import pickle
from src.config import MODEL_PATH
from src.data_processing import preprocess_data

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(input_data):
    model = load_model(MODEL_PATH)
    df = pd.DataFrame(input_data, index=[0])
    processed_df = preprocess_data(df)
    
    if processed_df.empty:
        return None

    probability = model.predict_proba(processed_df)[:, 1]
    return probability[0]


if __name__ == '__main__':
    new_client = {
        'id': 99999, 'age': 35, 'monthly_income': 7500, 'collateral_value': 60000,
        'loan_amount': 30000, 'city': 'Rio de Janeiro', 'state': 'RJ',
        'collateral_debt': 0.0, 'verified_restriction': False, 'dishonored_checks': 0,
        'expired_debts': 1, 'banking_debts': 1, 'commercial_debts': 0, 'protests': 0,
        'marital_status': 'MARRIED', 'informed_restriction': False, 'loan_term': 36.0,
        'monthly_payment': 1200.0, 'informed_purpose': 'OTHERS',
        'auto_brand': 'HONDA', 'auto_model': 'CIVIC', 'auto_year': 2018,
        'pre_approved': True, 'form_completed': False, 'sent_to_analysis': None, # O alvo é nulo
        'channel': 'ORGANIC_SEARCH', 'zip_code': '22222-222',
        'landing_page': 'home', 'landing_page_product': 'auto-loan',
        'gender': 'MALE', 'utm_term': None, 'education_level': 'POSTGRADUATE'
    }
    
    score = make_prediction(new_client)
    
    if score is not None:
        print("\nPREDIÇÃO PARA NOVO CLIENTE")
        print(f"\nProbabilidade de ser enviado para análise: {score:.2%}")