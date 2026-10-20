import os
import json
import numpy as np
import pandas as pd
import torch
from src.features import fit_baseline_curves, eval_curve
from src.claim_eval import forecast_stint_pace
from src.live import compute_cliff_life


def get_stint_sub(val_laps, stint_id):
    """Retrieve sorted laps for a stint."""
    return val_laps[val_laps["stint_id"] == stint_id].sort_values("stint_lap")


def get_predictions_at_lap(sub, curves, g_curve, lap, meds):
    """Compute frozen, rolling, always-20 and median forecasts for a stint."""
    f_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=1)
    r_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=lap - 1)
    f_c = compute_cliff_life(f_deg, 1.0)
    r_c = compute_cliff_life(r_deg, 1.0)

    f_rem = min(20.0, max(0.0, (2 + f_c) - lap)) if f_c is not None else 20.0
    r_rem = min(20.0, float(r_c)) if r_c is not None else 20.0
    cmp_name = sub["compound"].iloc[0]
    m_rem = max(0.0, meds.get(cmp_name, 12.0) - lap)

    f_uncap = float((2 + f_c) - lap) if f_c is not None else np.nan
    r_uncap = float(r_c) if r_c is not None else np.nan
    return f_rem, r_rem, m_rem, f_uncap, r_uncap


def compute_horizon_lap_error(sub, curves, g_curve, lap, h):
    """Compute lap-time absolute error for horizon h for frozen and rolling."""
    if len(sub) < lap + h:
        return None, None
    act_y = float(sub["lap_time_fc_s"].iloc[lap - 1 + h])
    track, cmp_name = sub["track_id"].iloc[0], sub["compound"].iloc[0]
    params = curves.get((track, cmp_name), g_curve)

    curr_age_r = int(sub["tyre_age"].iloc[lap - 1])
    off_r = sub["lap_time_fc_s"].iloc[lap - 1] - eval_curve(params, np.array([curr_age_r]))[0]
    pred_r = float(eval_curve(params, np.array([curr_age_r + h]))[0] + off_r)

    curr_age_f = int(sub["tyre_age"].iloc[1])
    off_f = sub["lap_time_fc_s"].iloc[1] - eval_curve(params, np.array([curr_age_f]))[0]
    pred_f = float(eval_curve(params, np.array([curr_age_f + (lap + h - 2)]))[0] + off_f)
    return abs(pred_f - act_y), abs(pred_r - act_y)


def run_stint_boot(diffs, n_boot=1000, seed=42):
    """Compute 95% CI with paired stint bootstrap."""
    np.random.seed(seed)
    valid = np.array([d for d in diffs if not np.isnan(d)])
    if len(valid) == 0:
        return 0.0, (0.0, 0.0)
    means = [float(np.mean(np.random.choice(valid, size=len(valid), replace=True))) for _ in range(n_boot)]
    return float(np.mean(means)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_cluster_boot(race_diff_map, n_boot=1000, seed=42):
    """Compute 95% CI with race cluster bootstrap."""
    np.random.seed(seed)
    races = list(race_diff_map.keys())
    means = []
    for _ in range(n_boot):
        sampled = np.random.choice(races, size=len(races), replace=True)
        pooled = [d for r in sampled for d in race_diff_map[r]]
        if len(pooled) > 0:
            means.append(float(np.mean(pooled)))
    if len(means) == 0:
        return 0.0, (0.0, 0.0)
    return float(np.mean(means)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def eval_uncensored_stints(uncensored, val_laps, curves, g_curve, lap, meds):
    """Evaluate PWE, horizon lap errors, and bootstrap CIs on uncensored stints."""
    pwe_f, pwe_r, pwe_20, pwe_m = [], [], [], []
    uncap_f, uncap_r = [], []
    h_errs = {1: ([], []), 5: ([], []), 10: ([], [])}
    race_diffs = {}

    for _, s in uncensored.iterrows():
        sub = get_stint_sub(val_laps, s["stint_id"])
        if len(sub) < lap:
            continue
        true_rem = max(0.0, float(s["life_laps"] - lap))
        f_rem, r_rem, m_rem, f_un, r_un = get_predictions_at_lap(sub, curves, g_curve, lap, meds)
        if true_rem <= 20.0:
            pwe_f.append(abs(f_rem - true_rem))
            pwe_r.append(abs(r_rem - true_rem))
            pwe_20.append(abs(20.0 - true_rem))
            pwe_m.append(abs(m_rem - true_rem))
            r_id = sub["race_id"].iloc[0]
            race_diffs.setdefault(r_id, []).append(abs(f_rem - true_rem) - abs(r_rem - true_rem))
        if not np.isnan(f_un) and not np.isnan(r_un):
            uncap_f.append(abs(f_un - true_rem))
            uncap_r.append(abs(r_un - true_rem))
        for h in [1, 5, 10]:
            ef, er = compute_horizon_lap_error(sub, curves, g_curve, lap, h)
            if ef is not None:
                h_errs[h][0].append(ef)
                h_errs[h][1].append(er)
    return pwe_f, pwe_r, pwe_20, pwe_m, uncap_f, uncap_r, h_errs, race_diffs


def eval_censored_stints(censored, val_laps, curves, g_curve, lap):
    """Compute survival-consistency rate on censored stints."""
    c_f, c_r, c_20 = [], [], []
    for _, s in censored.iterrows():
        sub = get_stint_sub(val_laps, s["stint_id"])
        if len(sub) < lap:
            continue
        obs_surv = float(len(sub) - lap)
        f_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=1)
        r_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=lap - 1)
        f_c = compute_cliff_life(f_deg, 1.0)
        r_c = compute_cliff_life(r_deg, 1.0)
        f_rem = min(20.0, max(0.0, (2 + f_c) - lap)) if f_c is not None else 20.0
        r_rem = min(20.0, float(r_c)) if r_c is not None else 20.0
        c_f.append(f_rem >= obs_surv)
        c_r.append(r_rem >= obs_surv)
        c_20.append(20.0 >= obs_surv)
    return len(c_r), float(np.mean(c_f)), float(np.mean(c_r)), float(np.mean(c_20))


def compute_lap_record(lap, uncensored, censored, val_laps, curves, g_curve, meds):
    """Generate complete comparison record for a single evaluation lap."""
    pf, pr, p20, pm, uf, ur, herr, r_diffs = eval_uncensored_stints(uncensored, val_laps, curves, g_curve, lap, meds)
    diffs = np.array(pf) - np.array(pr)
    mean_d, ci_stint = run_stint_boot(diffs)
    _, ci_cluster = run_cluster_boot(r_diffs)
    c_n, c_f, c_r, c_20 = eval_censored_stints(censored, val_laps, curves, g_curve, lap)
    crossed_diff = np.mean(np.array(pf) != np.array(pr)) if len(pf) > 0 else 0.0

    return {
        "stint_lap": lap, "effective_n": len(pf),
        "pwe_frozen_capped": round(float(np.mean(pf)), 2), "pwe_rolling_capped": round(float(np.mean(pr)), 2),
        "pwe_mode_e_capped": round(float(np.mean(pr)), 2), "pwe_always_20": round(float(np.mean(p20)), 2),
        "pwe_always_median": round(float(np.mean(pm)), 2),
        "pwe_frozen_uncapped": round(float(np.mean(uf)), 2) if len(uf) > 0 else np.nan,
        "pwe_rolling_uncapped": round(float(np.mean(ur)), 2) if len(ur) > 0 else np.nan,
        "rolling_advantage": round(float(np.mean(diffs)), 2),
        "ci_95_stint": f"[{ci_stint[0]:.2f}, {ci_stint[1]:.2f}]",
        "ci_95_cluster": f"[{ci_cluster[0]:.2f}, {ci_cluster[1]:.2f}]",
        "crossing_share_diff": round(float(crossed_diff), 3),
        "mae_h1_frozen": round(float(np.mean(herr[1][0])), 3), "mae_h1_rolling": round(float(np.mean(herr[1][1])), 3),
        "mae_h5_frozen": round(float(np.mean(herr[5][0])), 3), "mae_h5_rolling": round(float(np.mean(herr[5][1])), 3),
        "mae_h10_frozen": round(float(np.mean(herr[10][0])), 3), "mae_h10_rolling": round(float(np.mean(herr[10][1])), 3),
        "censored_n": c_n, "censored_rate_frozen": round(c_f, 3),
        "censored_rate_rolling": round(c_r, 3), "censored_rate_always20": round(c_20, 3)
    }


def main():
    """Run Part D evaluation across laps 3, 5, 7, 10 and save results."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    stints = pd.read_parquet("data/processed/stints.parquet")
    curves, g_curve = fit_baseline_curves(tr_laps)

    sc = tr_laps[["stint_id", "compound"]].drop_duplicates()
    ts = stints.merge(sc, on="stint_id")
    meds = ts[ts["stint_id"].isin(tr_laps["stint_id"]) & ~ts["censored"]].groupby("compound")["life_laps"].median().to_dict()

    va_stints = stints[stints["stint_id"].isin(va_laps["stint_id"].unique())]
    uncensored = va_stints[~va_stints["censored"]]
    censored = va_stints[va_stints["censored"]]

    records = [compute_lap_record(l, uncensored, censored, va_laps, curves, g_curve, meds) for l in [3, 5, 7, 10]]
    df = pd.DataFrame(records)
    os.makedirs("reports/overnight", exist_ok=True)
    out_path = "reports/overnight/exp9_results.csv"
    df.to_csv(out_path, index=False)
    print("Part D completed successfully! Results saved to:", out_path)
    print(df.to_string())


if __name__ == "__main__":
    main()
