import os
import numpy as np
import pandas as pd
from src.baselines import (
    fit_model_b_ridge,
    fit_model_c_xgboost,
    predict_model_a,
    predict_model,
)
from src.evaluate import (
    compute_mae,
    compute_censored_shares,
    run_baseline_evaluation,
)


def test_compute_mae_with_nans():
    """Verify MAE computation correctly ignores NaNs and computes absolute difference."""
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.5, 2.0, 10.0, 3.0])
    # |1.0 - 1.5| = 0.5; |2.0 - 2.0| = 0.0; |4.0 - 3.0| = 1.0 -> mean = 1.5 / 3 = 0.5
    mae = compute_mae(y_true, y_pred)
    assert np.isclose(mae, 0.5)


def test_baseline_models_fit_and_predict():
    """Verify Ridge and XGBoost fit on 7 features and produce valid predictions."""
    np.random.seed(42)
    x = np.random.randn(20, 7)
    y = 0.5 * x[:, 2] + 0.2 * x[:, 4] + np.random.normal(0, 0.1, 20)
    y[0] = np.nan

    ridge = fit_model_b_ridge(x, y)
    preds_b = predict_model(ridge, x)
    assert len(preds_b) == 20
    assert not np.isnan(preds_b).any()

    xgb_mod = fit_model_c_xgboost(x, y)
    preds_c = predict_model(xgb_mod, x)
    assert len(preds_c) == 20
    assert not np.isnan(preds_c).any()

    preds_a = predict_model_a(20)
    assert len(preds_a) == 20
    assert (preds_a == 0.0).all()


def test_censored_shares():
    """Verify calculation of censored proportions per compound."""
    laps_mock = pd.DataFrame({
        "stint_id": ["s1", "s2", "s3", "s4"],
        "compound": ["HARD", "HARD", "SOFT", "SOFT"]
    })
    stints_mock = pd.DataFrame({
        "stint_id": ["s1", "s2", "s3", "s4"],
        "censored": [False, True, True, True]
    })
    shares = compute_censored_shares(laps_mock, stints_mock)
    assert shares["HARD"] == 0.5
    assert shares["SOFT"] == 1.0
    assert shares["ALL"] == 0.75


def test_baseline_table_generation():
    """Verify baseline evaluation output table structure in reports directory."""
    laps_path = "data/processed/laps.parquet"
    stints_path = "data/processed/stints.parquet"
    assert os.path.exists(laps_path) and os.path.exists(stints_path)

    laps = pd.read_parquet(laps_path)
    stints = pd.read_parquet(stints_path)
    train_laps = laps[laps["race_id"] == "2023_01"]
    val_laps = laps[laps["race_id"] == "2023_02"]

    table_df = run_baseline_evaluation(train_laps, val_laps, stints)
    assert os.path.exists("reports/baseline_table.csv")
    assert "mae_h1" in table_df.columns
    assert "mae_h5" in table_df.columns
    assert "mae_h10" in table_df.columns
    assert "censored_share" in table_df.columns
    assert not table_df.empty
