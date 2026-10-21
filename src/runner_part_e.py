import os, json, numpy as np, pandas as pd, torch
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
            q_dict[t][0].append(float(np.percentile(lives, alpha * 100)))
            q_dict[t][1].append(float(np.percentile(lives, (1.0 - alpha) * 100)))

    res_q = {t: (np.array(q_dict[t][0]), np.array(q_dict[t][1])) for t in targets}
    return np.array(p50_list), np.array(true_list), res_q


def split_2024_val_races(val_laps, seed=42):
    """Split 2024 validation races in half by race with a fixed random seed."""
    races = sorted(val_laps["race_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = list(races)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def compute_conformal_quantiles(cal_p50, cal_true, targets):
    """Compute split-conformal absolute residual quantiles on calibration half."""
    cal_resids = np.abs(cal_true - cal_p50)
    n_cal = len(cal_resids)
    sorted_resids = np.sort(cal_resids)
    quantiles = {}
    for t in targets:
        k_idx = min(int(np.ceil((n_cal + 1) * t)), n_cal)
        quantiles[t] = float(sorted_resids[k_idx - 1])
    return quantiles


def evaluate_conformal_and_mc(eval_p50, eval_true, eval_q, quantiles, targets):
    """Evaluate conformal and raw MC-dropout coverage and width on evaluation half."""
    records = []
    eval_errs = np.abs(eval_true - eval_p50)
    for t in targets:
        q_val = quantiles[t]
        conf_cov = float(np.mean(eval_errs <= q_val))
        conf_width = round(2.0 * q_val, 2)
        mc_low, mc_high = eval_q[t]
        mc_cov = float(np.mean((eval_true >= mc_low) & (eval_true <= mc_high)))
        mc_width = round(float(np.mean(mc_high - mc_low)), 2)

        records.append({
            "target_coverage": t, "conformal_quantile_q": round(q_val, 2),
            "raw_mc_coverage": round(mc_cov, 3), "raw_mc_width": mc_width,
            "conformal_coverage": round(conf_cov, 3), "conformal_width": conf_width
        })
    return pd.DataFrame(records)


def save_conformal_results(df_cov, quantiles, cal_races, eval_races):
    """Save evaluation metrics to csv and conformal quantiles to json."""
    df_cov.to_csv("reports/uncertainty_coverage.csv", index=False)
    out_json = {str(k): v for k, v in quantiles.items()}
    out_json["cal_races"] = cal_races
    out_json["eval_races"] = eval_races
    with open("reports/conformal_quantiles.json", "w") as f:
        json.dump(out_json, f, indent=2)


def run_part_e():
    """Execute split-conformal calibration on 2024 races and save reports."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    stints = pd.read_parquet("data/processed/stints.parquet")
    curves, g_curve = fit_baseline_curves(tr_laps)
    track_map = {t: i for i, t in enumerate(tr_laps["track_id"].unique())}
    compound_map = {c: i for i, c in enumerate(tr_laps["compound"].unique())}

    ckpt = torch.load("runs/gru_main/checkpoint.pt", map_location="cpu", weights_only=False)
    model = create_gru_model()
    model.load_state_dict(ckpt)
    enable_dropout_at_inference(model)

    cal_races, eval_races = split_2024_val_races(va_laps, seed=42)
    cal_laps = va_laps[va_laps["race_id"].isin(cal_races)]
    eval_laps = va_laps[va_laps["race_id"].isin(eval_races)]
    cal_stints = stints[stints["stint_id"].isin(cal_laps["stint_id"].unique())]
    eval_stints = stints[stints["stint_id"].isin(eval_laps["stint_id"].unique())]

    targets = [0.50, 0.60, 0.70, 0.80, 0.90]
    cal_p50, cal_true, _ = collect_split_quantiles(model, cal_laps, cal_stints, curves, g_curve, track_map, compound_map, lap=5)
    quantiles = compute_conformal_quantiles(cal_p50, cal_true, targets)

    eval_p50, eval_true, eval_q = collect_split_quantiles(model, eval_laps, eval_stints, curves, g_curve, track_map, compound_map, lap=5)
    df_cov = evaluate_conformal_and_mc(eval_p50, eval_true, eval_q, quantiles, targets)
    save_conformal_results(df_cov, quantiles, cal_races, eval_races)
    print("Part 6 completed! Saved coverage and conformal_quantiles.json")
    print(df_cov.to_string())
    return df_cov


if __name__ == "__main__":
    run_part_e()
