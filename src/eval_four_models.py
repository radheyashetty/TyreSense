import os, numpy as np, pandas as pd
from src.app_data import load_cached_model, get_cached_maps
from src.app_forecast import forecast_future_laps, compute_live_k, compute_cliff_lap_int
from src.split import assert_no_2025_data


def get_eval_models_spec():
    """Return model evaluation specifications."""
    return [
        ("GRU", "GRU Forecaster (Primary Recurrent Net)", load_cached_model("GRU")),
        ("LSTM", "LSTM Forecaster (Cell-State Memory)", load_cached_model("LSTM")),
        ("Vanilla RNN", "Vanilla RNN (Baseline Recurrent Net)", load_cached_model("RNN")),
        ("Model E", "Neural-Guided Adaptive (Model E)", None),
        ("Baseline Curve", "Model A", None),
    ]


def evaluate_single_stint(sub, models_spec, curves, g_curve, track_map, compound_map, cutoff=5):
    """Compute 20-lap forecasts and cliff laps for all models on one stint at cutoff lap."""
    obs = sub[sub["stint_lap"] <= cutoff]
    last_obs = float(obs["lap_time_fc_s"].iloc[-1])
    k_val = compute_live_k(sub, cutoff, curves, g_curve)
    stint_id = sub["stint_id"].iloc[0]

    records = {}
    for name, full_name, m_obj in models_spec:
        f_laps, f_times, b_time = forecast_future_laps(
            obs, curves, g_curve, horizon=20, k=k_val,
            model_choice=full_name, model_obj=m_obj, track_map=track_map, compound_map=compound_map
        )
        cliff = compute_cliff_lap_int(f_laps, f_times, b_time)
        records[name] = {"times": f_times, "cliff": cliff}

    records["Persistence"] = {"times": np.full(20, last_obs), "cliff": None}
    return stint_id, records


def extract_stint_errors(records, sub, true_cliff, cutoff=5):
    """Calculate horizon errors and cliff PWE for each model on one stint."""
    out = []
    fut_actual = sub[sub["stint_lap"] > cutoff]
    for name, pred in records.items():
        row = {"model": name}
        for h in [1, 5, 10]:
            target_lap = cutoff + h
            match = sub[sub["stint_lap"] == target_lap]
            row[f"mae_h{h}"] = abs(pred["times"][h - 1] - float(match["lap_time_fc_s"].iloc[0])) if len(match) > 0 else np.nan

        if true_cliff is not None:
            l_pred = min(pred["cliff"], 22) if pred["cliff"] is not None else 22
            l_true = min(int(true_cliff), 22)
            row["pwe_abs"] = abs(l_pred - l_true)
        else:
            row["pwe_abs"] = np.nan
        out.append(row)
    return out


def run_four_model_validation(val_laps, stints_df, curves, g_curve, track_map, compound_map, cutoff=5):
    """Execute validation benchmark across all 2024 stints and aggregate metrics."""
    assert_no_2025_data(val_laps, "val_laps")
    models_spec = get_eval_models_spec()
    stint_map = {row["stint_id"]: row.to_dict() for _, row in stints_df.iterrows()}
    all_rows = []

    for sid in sorted(val_laps["stint_id"].unique()):
        sub = val_laps[val_laps["stint_id"] == sid].sort_values("stint_lap")
        if len(sub) <= cutoff:
            continue
        st_info = stint_map.get(sid, {})
        t_cliff = st_info.get("life_laps") if (st_info and not st_info.get("censored", True)) else None

        sid, recs = evaluate_single_stint(sub, models_spec, curves, g_curve, track_map, compound_map, cutoff)
        all_rows.extend(extract_stint_errors(recs, sub, t_cliff, cutoff))

    df = pd.DataFrame(all_rows)
    return df.groupby("model").agg(
        mae_h1=("mae_h1", "mean"), mae_h5=("mae_h5", "mean"),
        mae_h10=("mae_h10", "mean"), pwe_cliff=("pwe_abs", "mean"),
        n_stints=("mae_h1", "count")
    ).reset_index()


def main():
    """Load cached data and generate four-model validation summary table."""
    from src.app_data import load_app_data
    laps, stints, curves, g_curve = load_app_data()
    track_map, compound_map = get_cached_maps()
    val_laps = pd.read_parquet("data/processed/val_laps.parquet")

    summary = run_four_model_validation(val_laps, stints, curves, g_curve, track_map, compound_map, cutoff=5)
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/four_model_validation_summary.csv"
    summary.to_csv(out_path, index=False)
    print("Four-model validation summary saved to:", out_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
