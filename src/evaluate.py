import os
import numpy as np
import pandas as pd
from src.download import load_config
from src.features import (
    fit_baseline_curves,
    predict_causal_baseline,
    build_7_features,
    build_horizon_targets,
)
from src.baselines import (
    predict_model_a,
    predict_model,
    train_baselines_for_horizons,
)


def compute_mae(y_true, y_pred):
    """Calculate Mean Absolute Error ignoring NaN values."""
    valid_mask = ~np.isnan(y_true)
    if valid_mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask])))


def extract_features_and_targets(laps_df, curves, global_curve, track_map, compound_map):
    """Extract 7 features and horizon targets across all stints in a laps dataframe."""
    feature_list = []
    target_list = []
    stint_ids = laps_df["stint_id"].unique()

    for stint_id in stint_ids:
        sub = laps_df[laps_df["stint_id"] == stint_id].sort_values("stint_lap")
        feats = build_7_features(sub, track_map, compound_map)
        baseline = predict_causal_baseline(sub, curves, global_curve)
        targets = build_horizon_targets(sub, baseline, horizon=20)
        feature_list.append(feats)
        target_list.append(targets)

    if not feature_list:
        return np.empty((0, 8)), np.empty((0, 20))
    all_feats = pd.concat(feature_list, ignore_index=True)
    all_targets = np.vstack(target_list)
    return all_feats, all_targets


def compute_censored_shares(laps_df, stints_df):
    """Compute share of censored stints grouped by tyre compound."""
    stint_compounds = laps_df.groupby("stint_id")["compound"].first().reset_index()
    merged = pd.merge(stints_df, stint_compounds, on="stint_id", how="inner")
    shares = merged.groupby("compound")["censored"].mean().to_dict()
    shares["ALL"] = float(merged["censored"].mean())
    return shares


def evaluate_horizons(y_true_targets, y_pred_dict, compound_mask, horizons=(1, 5, 10)):
    """Evaluate MAE for horizons 1, 5, 10 on a subset masked by compound."""
    results = {}
    for h in horizons:
        y_true_h = y_true_targets[compound_mask, h - 1]
        y_pred_h = y_pred_dict[h][compound_mask]
        results[f"mae_h{h}"] = compute_mae(y_true_h, y_pred_h)
    return results


def compile_baseline_table(val_laps, val_targets, models, shares, horizons, compounds):
    """Build summary evaluation dataframe across models and tyre compounds."""
    table_rows = []
    compound_list = ["ALL"] + compounds
    for mod_name, mod_preds in models.items():
        for comp in compound_list:
            if comp == "ALL":
                mask = np.ones(len(val_laps), dtype=bool)
            else:
                mask = (val_laps["compound"].values == comp)
            if mask.sum() == 0:
                continue
            row = {
                "model": mod_name,
                "compound": comp,
                "censored_share": shares.get(comp, np.nan),
                "laps": int(mask.sum())
            }
            row.update(evaluate_horizons(val_targets, mod_preds, mask, horizons))
            table_rows.append(row)
    return pd.DataFrame(table_rows)


def prepare_and_train_baselines(train_laps, val_laps, horizons):
    """Extract 7 features and fit Ridge and XGBoost models for each horizon."""
    tracks = sorted(train_laps["track_id"].unique())
    track_map = {t: idx + 1 for idx, t in enumerate(tracks)}
    compounds = sorted(train_laps["compound"].unique())
    compound_map = {c: idx + 1 for idx, c in enumerate(compounds)}

    curves, global_curve = fit_baseline_curves(train_laps)
    train_feats, train_targets = extract_features_and_targets(
        train_laps, curves, global_curve, track_map, compound_map
    )
    val_feats, val_targets = extract_features_and_targets(
        val_laps, curves, global_curve, track_map, compound_map
    )

    feat_cols = [
        "track", "compound", "tyre_age", "lap_fraction",
        "relative_pace", "track_temp", "track_status"
    ]
    x_train = train_feats[feat_cols].values
    x_val = val_feats[feat_cols].values
    ridge_mods, xgb_mods = train_baselines_for_horizons(x_train, train_targets, horizons)

    preds = {
        "Model A (Baseline Curve)": {h: predict_model_a(len(val_feats)) for h in horizons},
        "Model B (Ridge)": {h: predict_model(ridge_mods[h], x_val) for h in horizons},
        "Model C (XGBoost)": {h: predict_model(xgb_mods[h], x_val) for h in horizons}
    }
    return val_targets, preds, compounds


def run_baseline_evaluation(train_laps, val_laps, stints_df, config_path="config.yaml"):
    """Train Models A, B, C and compute MAE at horizons 1, 5, 10 and censored shares."""
    config = load_config(config_path)
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    horizons = (1, 5, 10)

    val_targets, models, compounds = prepare_and_train_baselines(
        train_laps, val_laps, horizons
    )
    shares = compute_censored_shares(val_laps, stints_df)
    table_df = compile_baseline_table(
        val_laps, val_targets, models, shares, horizons, compounds
    )
    table_path = os.path.join(reports_dir, "baseline_table.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved baseline table to {table_path}")
    return table_df
