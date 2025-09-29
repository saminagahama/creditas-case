LIGHTGBM_OPTIMIZED_PARAMS = {
'scale_pos_weight': 10, 'n_estimators': 200, 'learning_rate': 0.028704795063264867, 'num_leaves': 26, 'max_depth': 15, 'subsample': 0.8111838798774171, 'colsample_bytree': 0.5511669788927203, 'reg_alpha': 0.002901554428894764, 'reg_lambda': 0.31368670121374115, 'min_child_samples': 93
}

ENSEMBLE_OPTIMIZED_PARAMS = {
    'positive_class_weight': 2.4263355281219705,
    'xgb_n_estimators': 237,
    'xgb_max_depth': 5,
    'xgb_learning_rate': 0.04002422297956846,
    'lgbm_n_estimators': 470,
    'lgbm_max_depth': 6,
    'lgbm_learning_rate': 0.020055380488421603,
    'lgbm_num_leaves': 57,
    'rf_n_estimators': 209,
    'rf_max_depth': 6,
    'lr_C': 2.8502467701841208,
    'mlp_hidden_layers': (50, 25),
    'mlp_alpha': 0.02003439087027942,
    'mlp_lr_init': 0.007554728728012091
}

def get_lightgbm_params():
    return LIGHTGBM_OPTIMIZED_PARAMS.copy()

def get_ensemble_params():
    return ENSEMBLE_OPTIMIZED_PARAMS.copy()