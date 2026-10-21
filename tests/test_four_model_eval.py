import numpy as np, pandas as pd
from src.app_data import load_app_data, get_cached_maps
from src.eval_four_models import evaluate_single_stint, extract_stint_errors, run_four_model_validation


def test_evaluate_single_stint_returns_all_models():
    """Verify evaluate_single_stint returns records for all 5 models + persistence."""
    laps, stints, curves, g_curve = load_app_data()
    track_map, compound_map = get_cached_maps()
    sub = laps[laps["stint_id"] == laps["stint_id"].iloc[0]].sort_values("stint_lap")

    from src.app_data import load_cached_model
    models_spec = [
        ("GRU", "GRU Forecaster", load_cached_model("GRU")),
        ("LSTM", "LSTM Forecaster", load_cached_model("LSTM")),
        ("Vanilla RNN", "Vanilla RNN", load_cached_model("RNN")),
        ("Model E", "Neural-Guided Adaptive (Model E)", None),
        ("Baseline Curve", "Model A", None),
    ]
    sid, recs = evaluate_single_stint(sub, models_spec, curves, g_curve, track_map, compound_map, cutoff=5)
    assert sid == sub["stint_id"].iloc[0]
    assert "GRU" in recs
    assert "LSTM" in recs
    assert "Vanilla RNN" in recs
    assert "Model E" in recs
    assert "Baseline Curve" in recs
    assert "Persistence" in recs

    for name, data in recs.items():
        assert len(data["times"]) == 20
        assert not np.isnan(data["times"]).any()


def test_extract_stint_errors_computes_mae_and_pwe():
    """Verify extract_stint_errors calculates finite horizon MAE and cliff PWE."""
    laps, stints, curves, g_curve = load_app_data()
    sub = laps[laps["stint_id"] == laps["stint_id"].iloc[0]].sort_values("stint_lap")
    records = {
        "Model A": {"times": np.full(20, 95.0), "cliff": 12},
        "Persistence": {"times": np.full(20, 94.0), "cliff": None},
    }
    rows = extract_stint_errors(records, sub, true_cliff=14.0, cutoff=5)
    assert len(rows) == 2
    row_a = [r for r in rows if r["model"] == "Model A"][0]
    assert not np.isnan(row_a["mae_h1"])
    assert row_a["pwe_abs"] == abs(12 - 14)  # 2.0


def test_run_four_model_validation_on_val_sample():
    """Verify run_four_model_validation produces aggregated dataframe on validation sample."""
    laps, stints, curves, g_curve = load_app_data()
    track_map, compound_map = get_cached_maps()
    val_laps = pd.read_parquet("data/processed/val_laps.parquet")
    sample_stints = val_laps["stint_id"].unique()[:5]
    sample_val = val_laps[val_laps["stint_id"].isin(sample_stints)]

    summary = run_four_model_validation(sample_val, stints, curves, g_curve, track_map, compound_map, cutoff=5)
    assert len(summary) >= 5
    models_found = summary["model"].tolist()
    assert "GRU" in models_found
    assert "LSTM" in models_found
    assert "Vanilla RNN" in models_found
    assert "Model E" in models_found
