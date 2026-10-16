import numpy as np
from sklearn.linear_model import Ridge
import xgboost as xgb


def fit_model_b_ridge(features_matrix, targets_vector, alpha_reg=1.0):
    """Fit Model B Ridge linear regression on valid non-nan horizon targets."""
    valid_mask = ~np.isnan(targets_vector)
    if valid_mask.sum() == 0:
        return None
    model = Ridge(alpha=alpha_reg)
    model.fit(features_matrix[valid_mask], targets_vector[valid_mask])
    return model


def fit_model_c_xgboost(features_matrix, targets_vector, seed=42):
    """Fit Model C XGBoost regressor on valid non-nan horizon targets."""
    valid_mask = ~np.isnan(targets_vector)
    if valid_mask.sum() == 0:
        return None
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=1
    )
    model.fit(features_matrix[valid_mask], targets_vector[valid_mask])
    return model


def predict_model_a(num_samples):
    """Model A baseline curve predicts zero residual over baseline."""
    return np.zeros(num_samples)


def predict_model(model, features_matrix):
    """Predict residuals using a trained scikit-learn or XGBoost model."""
    if model is None:
        return np.zeros(len(features_matrix))
    return model.predict(features_matrix)


def train_baselines_for_horizons(features_train, targets_train, horizons=(1, 5, 10)):
    """Train Ridge and XGBoost models for each specified forecast horizon."""
    ridge_models = {}
    xgb_models = {}
    for h in horizons:
        # Horizon h is at index h - 1
        target_col = targets_train[:, h - 1]
        ridge_models[h] = fit_model_b_ridge(features_train, target_col)
        xgb_models[h] = fit_model_c_xgboost(features_train, target_col)
    return ridge_models, xgb_models
