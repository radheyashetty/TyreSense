import os, json, pandas as pd, torch
from src.config import load_config
from src.features import fit_baseline_curves
from src.models import create_gru_model, create_lstm_model, create_rnn_model


def assert_no_2025_in_app(df):
    """Enforce test lock: app code must never load or expose 2025 data."""
    if "race_id" in df.columns:
        has_2025 = df["race_id"].astype(str).str.startswith("2025").any()
        if has_2025:
            raise AssertionError("R3 Violation: 2025 data detected in app context")


def load_app_data(config_path="config.yaml"):
    """Load cached processed laps, stints, curves, and conformal quantiles."""
    cfg = load_config(config_path)
    proc_dir = cfg["paths"]["processed_dir"]
    laps = pd.read_parquet(os.path.join(proc_dir, "laps.parquet"))
    stints = pd.read_parquet(os.path.join(proc_dir, "stints.parquet"))

    laps["season"] = laps["race_id"].str.slice(0, 4)
    laps = laps[laps["season"].isin(["2022", "2023", "2024"])].copy()
    assert_no_2025_in_app(laps)

    stints = stints[stints["stint_id"].isin(laps["stint_id"].unique())].copy()
    assert_no_2025_in_app(stints)

    train_laps = laps[laps["season"].isin(["2022", "2023"])]
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
    ckpt_map = {
        "GRU": ("runs/gru_main/checkpoint.pt", create_gru_model),
        "LSTM": ("runs/lstm_main/checkpoint.pt", create_lstm_model),
        "RNN": ("runs/rnn_main/checkpoint.pt", create_rnn_model),
    }
    if model_choice not in ckpt_map:
        return None
    path, factory = ckpt_map[model_choice]
    if not os.path.exists(path):
        return None
    model = factory(num_numeric=8)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def get_data_source_label(season):
    """Return factual data provider for given season."""
    return "FastF1" if season == "2022" else "OpenF1"


def get_data_limits_summary():
    """Read censored share and baseline comparison from reports."""
    exp9_path = "reports/overnight/exp9_results.csv"
    cens_share = "61.5%"
    if os.path.exists(exp9_path):
        df = pd.read_csv(exp9_path)
        if "censored_share" in df.columns:
            cens_share = f"{round(df['censored_share'].iloc[0] * 100, 1)}%"
    return cens_share
