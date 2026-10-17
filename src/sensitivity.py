import os
import pandas as pd
from src.fuel import fuel_correct, compute_fuel_coefficient
from src.download import load_config
from src.features import fit_baseline_curves
from src.evaluate import extract_features_and_targets, compute_mae
from src.baselines import train_baselines_for_horizons, predict_model


def get_coefficient(setting_name, track_id, config):
    """Retrieve fuel burn rate according to sensitivity setting name."""
    if setting_name == "track_scaled":
        return compute_fuel_coefficient(track_id, config)
    elif setting_name == "base_0.02":
        return 0.02
    elif setting_name == "base_0.04":
        return 0.04
    return 0.03


def run_one_setting(setting_name, tr_df, va_df, config, maps):
    """Evaluate Ridge and XGBoost for a specific fuel coefficient setting."""
    tmap, cmap = maps
    tr = tr_df.copy()
    va = va_df.copy()
    c_tr = get_coefficient(setting_name, tr["track_id"].iloc[0], config)
    c_va = get_coefficient(setting_name, va["track_id"].iloc[0], config)
    tr["lap_time_fc_s"] = fuel_correct(tr["lap_time_s"], tr["lap_number"], tr["total_laps"], c=c_tr)
    va["lap_time_fc_s"] = fuel_correct(va["lap_time_s"], va["lap_number"], va["total_laps"], c=c_va)

    curves, g_curve = fit_baseline_curves(tr)
    tr_f, tr_t = extract_features_and_targets(tr, curves, g_curve, tmap, cmap)
    va_f, va_t = extract_features_and_targets(va, curves, g_curve, tmap, cmap)
    fcols = ["track", "compound", "tyre_age", "lap_fraction", "relative_pace", "track_temp", "track_status"]
    x_tr = tr_f[fcols].values
    x_va = va_f[fcols].values

    ridge_mods, xgb_mods = train_baselines_for_horizons(x_tr, tr_t, (1, 5, 10))
    rows = []
    for m_name, mods in [("Ridge", ridge_mods), ("XGBoost", xgb_mods)]:
        m1 = compute_mae(va_t[:, 0], predict_model(mods[1], x_va))
        m5 = compute_mae(va_t[:, 4], predict_model(mods[5], x_va))
        m10 = compute_mae(va_t[:, 9], predict_model(mods[10], x_va))
        rows.append({"setting": setting_name, "model": m_name, "mae_h1": round(m1, 3), "mae_h5": round(m5, 3), "mae_h10": round(m10, 3)})
    return rows


def run_fuel_sensitivity(config_path="config.yaml"):
    """Run sensitivity analysis for fuel coefficients and save reports/fuel_sensitivity.csv."""
    cfg = load_config(config_path)
    proc_dir = cfg["paths"]["processed_dir"]
    rep_dir = cfg["paths"]["reports_dir"]
    laps = pd.read_parquet(os.path.join(proc_dir, "laps.parquet"))
    tr = laps[laps["race_id"] == "2023_01"]
    va = laps[laps["race_id"] == "2023_02"]

    tmap = {t: i + 1 for i, t in enumerate(sorted(tr["track_id"].unique()))}
    cmap = {c: i + 1 for i, c in enumerate(sorted(tr["compound"].unique()))}
    settings = ["constant_0.03", "track_scaled", "base_0.02", "base_0.04"]

    records = []
    for s in settings:
        records.extend(run_one_setting(s, tr, va, cfg, (tmap, cmap)))

    out_df = pd.DataFrame(records)
    out_path = os.path.join(rep_dir, "fuel_sensitivity.csv")
    out_df.to_csv(out_path, index=False)
    print(out_df)
    return out_df
