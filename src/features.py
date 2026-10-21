import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def fit_quadratic_curve(tyre_ages, lap_times, alpha_reg=1.0):
    """Fit a 2nd degree polynomial ridge regression on tyre ages."""
    x_feat = np.column_stack([tyre_ages, tyre_ages ** 2])
    model = Ridge(alpha=alpha_reg)
    model.fit(x_feat, lap_times)
    return float(model.intercept_), float(model.coef_[0]), float(model.coef_[1])


def fit_baseline_curves(train_df, alpha_reg=1.0):
    """Fit quadratic tyre degradation curves per track and compound with global fallback."""
    curves = {}
    grouped = train_df.groupby(["track_id", "compound"])
    for (track, compound), group in grouped:
        if len(group) >= 5:
            curves[(track, compound)] = fit_quadratic_curve(
                group["tyre_age"].values, group["lap_time_fc_s"].values, alpha_reg
            )

    global_curve = fit_quadratic_curve(
        train_df["tyre_age"].values, train_df["lap_time_fc_s"].values, alpha_reg
    )
    return curves, global_curve


def eval_curve(curve_params, tyre_ages):
    """Evaluate quadratic polynomial coefficients at given tyre ages."""
    c0, c1, c2 = curve_params
    return c0 + c1 * tyre_ages + c2 * (tyre_ages ** 2)


def predict_causal_baseline(stint_df, curves, global_curve, causal_lap=3):
    """Compute causal baseline prediction using track curve before lap 3 and stint shift after."""
    track = stint_df["track_id"].iloc[0]
    compound = stint_df["compound"].iloc[0]
    params = curves.get((track, compound), global_curve)
    tyre_ages = stint_df["tyre_age"].values
    raw_curve = eval_curve(params, tyre_ages)
    baseline_preds = raw_curve.copy()

    stint_laps = stint_df["stint_lap"].values
    early_mask = stint_laps <= causal_lap
    if early_mask.sum() > 0:
        early_obs_mean = float(stint_df.loc[early_mask, "lap_time_fc_s"].mean())
        early_curve_mean = float(raw_curve[early_mask].mean())
        offset = early_obs_mean - early_curve_mean
        later_mask = stint_laps >= causal_lap
        baseline_preds[later_mask] = raw_curve[later_mask] + offset

    return baseline_preds


def compute_relative_pace(stint_df, causal_lap=3):
    """Calculate causal pace relative to the driver's own early laps."""
    fc_times = stint_df["lap_time_fc_s"].values
    stint_laps = stint_df["stint_lap"].values
    relative_pace = np.zeros(len(stint_df))
    early_mask = stint_laps <= causal_lap
    ref_pace = float(fc_times[early_mask].mean()) if early_mask.sum() > 0 else fc_times[0]

    for i in range(len(stint_df)):
        curr_ref = float(np.mean(fc_times[: i + 1])) if stint_laps[i] < causal_lap else ref_pace
        relative_pace[i] = fc_times[i] - curr_ref

    return relative_pace


def compute_temp_delta(stint_df):
    """Calculate causal track temperature delta relative to stint start."""
    start_temp = float(stint_df["track_temp_c"].iloc[0])
    return stint_df["track_temp_c"].values - start_temp


def build_7_features(stint_df, track_map, compound_map, causal_lap=3):
    """Construct the 7 specified model inputs and gap mask for a stint."""
    track_code = track_map.get(stint_df["track_id"].iloc[0], 0)
    features = pd.DataFrame()
    features["track"] = [track_code] * len(stint_df)
    features["compound"] = [compound_map.get(c, 0) for c in stint_df["compound"]]
    features["tyre_age"] = stint_df["tyre_age"].values
    features["lap_fraction"] = stint_df["lap_number"].values / stint_df["total_laps"].values
    features["relative_pace"] = compute_relative_pace(stint_df, causal_lap)
    features["track_temp"] = stint_df["track_temp_c"].values
    features["track_status"] = pd.to_numeric(stint_df["track_status"], errors="coerce").fillna(1).astype(int).values
    features["gap_mask"] = stint_df["gap_mask"].values
    return features


def build_stint_features(stint_df, track_map, compound_map, causal_lap=3, include_temp_delta=False):
    """Construct model inputs (7 or 8) and gap mask for a stint."""
    features = build_7_features(stint_df, track_map, compound_map, causal_lap)
    if include_temp_delta:
        features["track_temp_delta_c"] = compute_temp_delta(stint_df)
    return features


def build_horizon_targets(stint_df, baseline_preds, horizon=20):
    """Compute target residuals for future horizons 1 to horizon laps ahead."""
    residuals = stint_df["lap_time_fc_s"].values - baseline_preds
    num_laps = len(stint_df)
    targets = np.full((num_laps, horizon), np.nan)

    for i in range(num_laps):
        for h in range(1, horizon + 1):
            target_idx = i + h
            if target_idx < num_laps:
                targets[i, h - 1] = residuals[target_idx]
    return targets
