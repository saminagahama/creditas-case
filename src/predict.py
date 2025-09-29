import pandas as pd
import pickle
from src.config import MODEL_PATH, FEATURES_TO_EXCLUDE, TARGET, DEFAULT_THRESHOLD
from src.data_processing import preprocess_data

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(input_data, threshold=None):
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
        
    model = load_model(MODEL_PATH)
    df = pd.DataFrame(input_data, index=[0])
    processed_df = preprocess_data(df)
    processed_df.drop(columns=FEATURES_TO_EXCLUDE + [TARGET], inplace=True, errors='ignore')
    
    if processed_df.empty:
        return None

    probability = model.predict_proba(processed_df)[:, 1][0]
    prediction = probability >= threshold
    
    return {
        'probability': probability,
        'prediction': prediction,
        'threshold_used': threshold
    }

def predict_batch(data, threshold=None):
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
        
    model = load_model(MODEL_PATH)
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    processed_df = preprocess_data(df)
    processed_df.drop(columns=FEATURES_TO_EXCLUDE + [TARGET], inplace=True, errors='ignore')
    
    if processed_df.empty:
        return None
    
    probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = probabilities >= threshold
    
    results_df = pd.DataFrame({
        'probability': probabilities,
        'prediction': predictions,
        'threshold_used': threshold
    })
    
    return results_df


if __name__ == '__main__':
    import sys
    
    new_client = {
        'id': 12345, 'age': 28, 'monthly_income': 4500, 'collateral_value': 35000,
    'loan_amount': 15000, 'city': 'São Paulo', 'state': 'SP',
    'collateral_debt': 0.0, 'verified_restriction': False, 'dishonored_checks': 0,
    'expired_debts': 0, 'banking_debts': 0, 'commercial_debts': 0, 'protests': 0,
    'marital_status': 'SINGLE', 'informed_restriction': False, 'loan_term': 48.0,
    'monthly_payment': 800.0, 'informed_purpose': 'DEBT_CONSOLIDATION',
    'auto_brand': 'VOLKSWAGEN', 'auto_model': 'GOL', 'auto_year': 2016,
    'pre_approved': True, 'form_completed': True, 'sent_to_analysis': None,
    'channel': 'PAID_SEARCH', 'zip_code': '01000-000',
    'landing_page': 'campaign', 'landing_page_product': 'auto-loan',
    'gender': 'FEMALE', 'utm_term': 'emprestimo carro', 'education_level': 'HIGH_SCHOOL'
    }
    
    threshold = DEFAULT_THRESHOLD
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
            if threshold < 0 or threshold > 1:
                raise ValueError("Threshold deve estar entre 0 e 1")
        except ValueError as e:
            print(f"Erro: {e}")
            print("Uso: python -m src.predict [threshold]")
            print("Exemplo: python -m src.predict 0.3")
            sys.exit(1)
    
    result = make_prediction(new_client, threshold=threshold)
    
    if result is not None:
        print("\n=== PREDIÇÃO PARA NOVO CLIENTE ===")
        print(f"Probabilidade: {result['probability']:.2%}")
        print(f"Threshold usado: {result['threshold_used']:.2f}")
        print(f"Predição: {'ENVIAR PARA ANÁLISE' if result['prediction'] else 'NÃO ENVIAR'}")
        
        print("\n=== COMPARAÇÃO COM DIFERENTES THRESHOLDS ===")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            res = make_prediction(new_client, threshold=t)
            status = "Enviar" if res['prediction'] else "Não enviar"
            print(f"Threshold {t:.1f}: {status} (prob: {res['probability']:.2%})")