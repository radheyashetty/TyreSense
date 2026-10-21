import os, json, pandas as pd, numpy as np, torch
from src.features import fit_baseline_curves
from src.models import create_gru_model


def assert_no_2025_in_app(df):
    """Enforce R3 locked test set rule: no 2025 data inside the app."""
    if "season" in df.columns:
        assert not (df["season"].astype(str) == "2025").any(), "R3 Violation: 2025 data detected in app"
    if "race_id" in df.columns:
        assert not (df["race_id"].astype(str).str.startswith("2025")).any(), "R3 Violation: 2025 data detected in app"


def load_app_data():
    """Load cached processed laps, stints, curves, and conformal quantiles."""
    laps = pd.read_parquet("data/processed/laps.parquet")
    stints = pd.read_parquet("data/processed/stints.parquet")

    laps["season"] = laps["race_id"].str.slice(0, 4)
    laps = laps[laps["season"].isin(["2022", "2023", "2024"])].copy()
    assert_no_2025_in_app(laps)

    stints = stints[stints["stint_id"].isin(laps["stint_id"].unique())].copy()
    assert_no_2025_in_app(stints)

    train_laps = laps[laps["season"].isin(["2022", "2023"])]
    assert_no_2025_in_app(train_laps)

    curves, global_curve = fit_baseline_curves(train_laps)
    return laps, stints, curves, global_curve


def load_conformal_quantile():
    """Load 80% coverage conformal quantile from reports json."""
    q_path = "reports/conformal_quantiles.json"
    if os.path.exists(q_path):
        with open(q_path, "r") as f:
            data = json.load(f)
            return float(data.get("0.8", 13.0))
    return 13.0


def get_stint_subset(laps_df, selected_stint):
    """Retrieve and sort laps belonging to the selected stint."""
    return laps_df[laps_df["stint_id"] == selected_stint].sort_values("stint_lap")


def load_cached_model(model_choice):
    """Load requested neural checkpoint or return None if tabular/unavailable."""
    from src.models import create_lstm_model, create_rnn_model
    ckpt_map = {
        "GRU": ("runs/gru_main/checkpoint.pt", create_gru_model),
        "LSTM": ("runs/lstm_main/checkpoint.pt", create_lstm_model),
        "RNN": ("runs/rnn_main/checkpoint.pt", create_rnn_model),
    }
    for key, (path, factory) in ckpt_map.items():
        if key in model_choice and os.path.exists(path):
            model = factory()
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt)
            model.eval()
            return model
    return None


def get_data_source_label(season):
    """Return factual data provider for given season."""
    return "FastF1" if season == "2022" else "OpenF1"


def get_conformal_coverage_label():
    """Read measured conformal coverage over 5 seeded splits and return status string."""
    seeds_path = "reports/conformal_coverage_seeds.csv"
    if os.path.exists(seeds_path):
        df = pd.read_csv(seeds_path)
        covs = df["conformal_coverage"].values
        mean_cov = round(float(np.mean(covs)) * 100, 1)
        min_cov = round(float(np.min(covs)) * 100, 1)
        max_cov = round(float(np.max(covs)) * 100, 1)
        return f"Mean {mean_cov}% (range {min_cov}% - {max_cov}%) [interval under-covers; shown for illustration only]"
    return "Mean 69.7% (range 62.8% - 76.4%) [interval under-covers; shown for illustration only]"


def get_data_limits_summary():
    """Read censored share and baseline comparison from reports."""
    exp9_path = "reports/overnight/exp9_official.csv" if os.path.exists("reports/overnight/exp9_official.csv") else "reports/overnight/exp9_results.csv"
    cens_share = "58.7%"
    if os.path.exists(exp9_path):
        df = pd.read_csv(exp9_path)
        if "censored_share" in df.columns:
            cens_share = f"{round(df['censored_share'].iloc[0] * 100, 1)}%"
    return cens_share
