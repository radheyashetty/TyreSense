import os
import numpy as np
import pandas as pd
import torch
from src.models import create_gru_model
from src.features import fit_baseline_curves, predict_causal_baseline, build_stint_features, eval_curve
from src.live import compute_cliff_life
from src.uncertainty import enable_dropout_at_inference


def extract_stint_tensors(sub, lap, track_map, compound_map, curves, g_curve, passes=100):
    """Prepare batched tensors for MC-Dropout forward passes at evaluation lap."""
    feats = build_stint_features(sub, track_map, compound_map, include_temp_delta=True)
    baseline = predict_causal_baseline(sub, curves, g_curve)
    curr_res = (sub["lap_time_fc_s"].values - baseline).reshape(-1, 1)
    num_cols = ["tyre_age", "lap_fraction", "relative_pace", "track_temp", "track_status", "gap_mask", "track_temp_delta_c"]
    full_numeric = np.hstack([feats[num_cols].values, curr_res])

    seq_len = 10
    x_pad = np.zeros((seq_len, 8), dtype=np.float32)
    trk_pad = np.zeros(seq_len, dtype=np.int64)
    cmp_pad = np.zeros(seq_len, dtype=np.int64)

    obs_len = min(lap, seq_len)
    x_pad[-obs_len:] = full_numeric[lap - obs_len: lap]
    trk_pad[-obs_len:] = feats["track"].values[lap - obs_len: lap]
    cmp_pad[-obs_len:] = feats["compound"].values[lap - obs_len: lap]

    x_tensor = torch.tensor(x_pad).unsqueeze(0).repeat(passes, 1, 1).float()
    trk_tensor = torch.tensor(trk_pad).unsqueeze(0).repeat(passes, 1).long()
    cmp_tensor = torch.tensor(cmp_pad).unsqueeze(0).repeat(passes, 1).long()
    return x_tensor, trk_tensor, cmp_tensor


def sample_stint_mc_lives(model, sub, lap, track_map, compound_map, curves, g_curve, passes=100):
    """Run MC-Dropout passes and return sampled usable remaining lives."""
    x_t, trk_t, cmp_t = extract_stint_tensors(sub, lap, track_map, compound_map, curves, g_curve, passes)
    with torch.no_grad():
        res_preds = model(x_t, trk_t, cmp_t).numpy()

    track, cmp_name = sub["track_id"].iloc[0], sub["compound"].iloc[0]
    params = curves.get((track, cmp_name), g_curve)
    curr_age = int(sub["tyre_age"].iloc[lap - 1])
    base_preds = eval_curve(params, np.arange(curr_age + 1, curr_age + 21))
    early_laps = sub[sub["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())

    lives = []
    for p in range(passes):
        deg = base_preds + res_preds[p] - baseline_time
        cl = compute_cliff_life(deg, threshold_s=1.0)
        lives.append(float(cl) if cl is not None else 20.0)
    return np.array(lives)


def collect_split_quantiles(model, laps_df, stints_df, curves, g_curve, track_map, compound_map, lap=5):
    """Collect median and interval quantiles across uncensored stints for target coverages."""
    uncensored = stints_df[~stints_df["censored"]]
    targets = [0.50, 0.60, 0.70, 0.80, 0.90]
    p50_list, true_list = [], []
    q_dict = {t: ([], []) for t in targets}

    for _, s in uncensored.iterrows():
        sub = laps_df[laps_df["stint_id"] == s["stint_id"]].sort_values("stint_lap")
        if len(sub) < lap:
            continue
        true_rem = max(0.0, float(s["life_laps"] - lap))
        lives = sample_stint_mc_lives(model, sub, lap, track_map, compound_map, curves, g_curve)
        med = float(np.median(lives))
        p50_list.append(med)
        true_list.append(true_rem)
        for t in targets:
            alpha = (1.0 - t) / 2.0
            q_low = float(np.percentile(lives, alpha * 100))
            q_high = float(np.percentile(lives, (1.0 - alpha) * 100))
            q_dict[t][0].append(q_low)
            q_dict[t][1].append(q_high)

    res_q = {t: (np.array(q_dict[t][0]), np.array(q_dict[t][1])) for t in targets}
    return np.array(p50_list), np.array(true_list), res_q


def fit_scale_factors(train_p50, train_true, train_q, targets):
    """Fit interval scaling factor s on train uncensored stints for each target coverage."""
    scale_factors = {}
    for t in targets:
        q_low, q_high = train_q[t]
        best_s, best_diff = 1.0, float("inf")
        for s in np.linspace(0.1, 4.0, 80):
            sc_low = train_p50 + s * (q_low - train_p50)
            sc_high = train_p50 + s * (q_high - train_p50)
            cov = float(np.mean((train_true >= sc_low) & (train_true <= sc_high)))
            diff = abs(cov - t)
            if diff < best_diff:
                best_diff, best_s = diff, float(s)
        scale_factors[t] = best_s
    return scale_factors


def evaluate_scaled_coverage(val_p50, val_true, val_q, scale_factors, targets):
    """Evaluate raw and scaled coverage on validation uncensored stints."""
    records = []
    for t in targets:
        q_low, q_high = val_q[t]
        s = scale_factors[t]
        sc_low = val_p50 + s * (q_low - val_p50)
        sc_high = val_p50 + s * (q_high - val_p50)

        raw_cov = float(np.mean((val_true >= q_low) & (val_true <= q_high)))
        raw_width = float(np.mean(q_high - q_low))
        sc_cov = float(np.mean((val_true >= sc_low) & (val_true <= sc_high)))
        sc_width = float(np.mean(sc_high - sc_low))

        records.append({
            "target_coverage": t, "scale_factor_s": round(s, 2),
            "raw_coverage": round(raw_cov, 3), "raw_width": round(raw_width, 2),
            "scaled_coverage": round(sc_cov, 3), "scaled_width": round(sc_width, 2)
        })
    return pd.DataFrame(records)


def run_part_e():
    """Execute complete Part E uncertainty calibration and write reports/uncertainty_coverage.csv."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    stints = pd.read_parquet("data/processed/stints.parquet")
    curves, g_curve = fit_baseline_curves(tr_laps)

    track_map = {t: i for i, t in enumerate(tr_laps["track_id"].unique())}
    compound_map = {c: i for i, c in enumerate(tr_laps["compound"].unique())}

    ckpt = torch.load("runs/final_gru/checkpoint.pt", map_location="cpu", weights_only=False)
    model = create_gru_model()
    model.load_state_dict(ckpt)
    enable_dropout_at_inference(model)

    tr_stints = stints[stints["stint_id"].isin(tr_laps["stint_id"].unique())]
    va_stints = stints[stints["stint_id"].isin(va_laps["stint_id"].unique())]
    targets = [0.50, 0.60, 0.70, 0.80, 0.90]

    tr_p50, tr_true, tr_q = collect_split_quantiles(model, tr_laps, tr_stints, curves, g_curve, track_map, compound_map, lap=5)
    scale_factors = fit_scale_factors(tr_p50, tr_true, tr_q, targets)

    va_p50, va_true, va_q = collect_split_quantiles(model, va_laps, va_stints, curves, g_curve, track_map, compound_map, lap=5)
    df_cov = evaluate_scaled_coverage(va_p50, va_true, va_q, scale_factors, targets)

    out_path = "reports/uncertainty_coverage.csv"
    df_cov.to_csv(out_path, index=False)
    print("Part E completed! Saved to:", out_path)
    print(df_cov.to_string())
    return df_cov


if __name__ == "__main__":
    run_part_e()
