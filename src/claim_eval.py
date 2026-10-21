import os
import numpy as np
import pandas as pd
from src.download import load_config
from src.features import fit_baseline_curves, predict_causal_baseline, eval_curve
from src.live import compute_cliff_life, update_k_factor
from src.clean import flag_spikes_causal


def compute_pwe(pred_life, actual_life, max_horizon=20):
    """Compute Pit Window Error in laps, capping non-crossing forecasts at max_horizon."""
    if actual_life is None or pd.isna(actual_life):
        return np.nan
    eff_pred = max_horizon if (pred_life is None or pd.isna(pred_life)) else float(pred_life)
    return abs(eff_pred - float(actual_life))


def forecast_stint_pace(stint_df, curves, global_curve, lap_idx, horizon=20):
    """Generate forward residual and degradation forecast from observed laps."""
    track = stint_df["track_id"].iloc[0]
    compound = stint_df["compound"].iloc[0]
    params = curves.get((track, compound), global_curve)

    obs = stint_df.iloc[: lap_idx + 1]
    curr_age = int(obs["tyre_age"].iloc[-1])
    fut_ages = np.arange(curr_age + 1, curr_age + horizon + 1)
    base_preds = eval_curve(params, fut_ages)

    early_laps = obs[obs["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())
    offset = float(obs["lap_time_fc_s"].iloc[-1] - eval_curve(params, np.array([curr_age]))[0])
    forecast_fc = base_preds + offset
    forecast_deg = forecast_fc - baseline_time
    return forecast_deg


def simulate_modes_at_lap(stint_df, curves, global_curve, target_lap, threshold_s=1.0):
    """Evaluate FROZEN, ROLLING, and Model E at a specific stint lap."""
    num_laps = len(stint_df)
    if target_lap > num_laps:
        return None

    # FROZEN mode: forecast fixed at lap 2
    froz_deg = forecast_stint_pace(stint_df, curves, global_curve, lap_idx=1)
    froz_life = compute_cliff_life(froz_deg, threshold_s)

    # ROLLING mode: forecast recomputed at target_lap
    roll_deg = forecast_stint_pace(stint_df, curves, global_curve, lap_idx=target_lap - 1)
    roll_life = compute_cliff_life(roll_deg, threshold_s)

    # Model E: online k-update up to target_lap
    k, k_hist = 1.0, []
    base_preds = predict_causal_baseline(stint_df, curves, global_curve)
    for i in range(1, target_lap):
        s_lap = int(stint_df["stint_lap"].iloc[i])
        act_t = float(stint_df["lap_time_fc_s"].iloc[i])
        prev_act = float(stint_df["lap_time_fc_s"].iloc[i - 1])
        delta_b = float(base_preds[i] - base_preds[i - 1])
        spike = flag_spikes_causal(stint_df.iloc[: i + 1])
        if s_lap >= 3:
            k, k_hist = update_k_factor(k, k_hist, act_t, prev_act, delta_b, spike)

    mode_e_deg = k * roll_deg
    mode_e_life = compute_cliff_life(mode_e_deg, threshold_s)
    return {"frozen": froz_life, "rolling": roll_life, "mode_e": mode_e_life}


def run_paired_bootstrap(diffs, n_boot=1000, seed=42):
    """Compute paired bootstrap mean, standard deviation, and 95% confidence interval."""
    np.random.seed(seed)
    valid_diffs = diffs[~np.isnan(diffs)]
    if len(valid_diffs) == 0:
        return 0.0, 0.0, (0.0, 0.0)
    boot_means = [np.mean(np.random.choice(valid_diffs, size=len(valid_diffs), replace=True)) for _ in range(n_boot)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return float(np.mean(boot_means)), float(np.std(boot_means)), (ci_low, ci_high)


def score_censored_stints(censored_df, laps_df, curves, global_curve, lap, threshold_s=1.0):
    """Score concordance for censored stints (fraction predicting survival past observed laps)."""
    cens_roll, cens_froz = [], []
    for _, row in censored_df.iterrows():
        stint_id = row["stint_id"]
        sub = laps_df[laps_df["stint_id"] == stint_id].sort_values("stint_lap")
        preds = simulate_modes_at_lap(sub, curves, global_curve, target_lap=lap, threshold_s=threshold_s)
        if preds is not None:
            stint_len = float(len(sub))
            r_life = 20.0 if preds["rolling"] is None else float(preds["rolling"])
            f_life = 20.0 if preds["frozen"] is None else float(preds["frozen"])
            cens_roll.append(r_life >= stint_len)
            cens_froz.append(f_life >= stint_len)
    cr = float(np.mean(cens_roll)) if cens_roll else np.nan
    cf = float(np.mean(cens_froz)) if cens_froz else np.nan
    return len(cens_roll), cr, cf


def build_exp9_record(lap, froz_arr, roll_arr, e_arr, mean_diff, ci, c_n, c_roll, c_froz):
    """Format single stint lap evaluation summary record."""
    ci_l, ci_h = ci
    return {
        "stint_lap": lap, "effective_n": int(len(froz_arr)),
        "pwe_frozen": round(float(np.nanmean(froz_arr)), 2),
        "pwe_rolling": round(float(np.nanmean(roll_arr)), 2),
        "pwe_mode_e": round(float(np.nanmean(e_arr)), 2),
        "share_within_2_rolling": round(float(np.mean(roll_arr <= 2.0)), 3),
        "share_within_2_mode_e": round(float(np.mean(e_arr <= 2.0)), 3),
        "rolling_vs_frozen_diff": round(mean_diff, 2),
        "ci_95": f"[{ci_l:.2f}, {ci_h:.2f}]",
        "censored_n": c_n,
        "censored_concordance_rolling": round(c_roll, 3) if not np.isnan(c_roll) else np.nan,
        "censored_concordance_frozen": round(c_froz, 3) if not np.isnan(c_froz) else np.nan
    }


def evaluate_exp9_stints(stints_df, laps_df, curves, global_curve, eval_laps=(3, 5, 7), threshold_s=1.0):
    """Evaluate PWE and 2-lap accuracy across stints for EXP-9 reporting effective N."""
    records = []
    uncensored = stints_df[stints_df["censored"] == False]
    censored = stints_df[stints_df["censored"] == True]

    for lap in eval_laps:
        pwe_froz, pwe_roll, pwe_e = [], [], []
        for _, stint_row in uncensored.iterrows():
            stint_id = stint_row["stint_id"]
            true_life = stint_row["life_laps"]
            sub = laps_df[laps_df["stint_id"] == stint_id].sort_values("stint_lap")
            preds = simulate_modes_at_lap(sub, curves, global_curve, target_lap=lap, threshold_s=threshold_s)
            if preds is not None:
                pwe_froz.append(compute_pwe(preds["frozen"], true_life))
                pwe_roll.append(compute_pwe(preds["rolling"], true_life))
                pwe_e.append(compute_pwe(preds["mode_e"], true_life))

        froz_arr, roll_arr, e_arr = np.array(pwe_froz), np.array(pwe_roll), np.array(pwe_e)
        mean_diff, _, ci = run_paired_bootstrap(froz_arr - roll_arr)
        c_n, c_roll, c_froz = score_censored_stints(censored, laps_df, curves, global_curve, lap, threshold_s)
        records.append(build_exp9_record(lap, froz_arr, roll_arr, e_arr, mean_diff, ci, c_n, c_roll, c_froz))
    return pd.DataFrame(records)


def run_exp9(config_path="config.yaml"):
    """Execute EXP-9 core claim benchmark comparing FROZEN, ROLLING, and Model E."""
    cfg = load_config(config_path)
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    stints = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "stints.parquet"))
    curves, global_curve = fit_baseline_curves(laps)

    df_exp9 = evaluate_exp9_stints(stints, laps, curves, global_curve)
    rep_dir = cfg["paths"]["reports_dir"]
    out_path = os.path.join(rep_dir, "exp9_core_claim.csv")
    df_exp9.to_csv(out_path, index=False)
    print("EXP-9 Core Claim Results:")
    print(df_exp9)
    return df_exp9


def run_cliff_sensitivity(config_path="config.yaml", thresholds=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """Evaluate Model E across different degradation cliff definitions."""
    cfg = load_config(config_path)
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    stints = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "stints.parquet"))
    curves, global_curve = fit_baseline_curves(laps)

    rows = []
    for th in thresholds:
        df_th = evaluate_exp9_stints(stints, laps, curves, global_curve, eval_laps=(5,), threshold_s=th)
        rows.append({
            "threshold_s": th,
            "pwe_mode_e_lap5": df_th["pwe_mode_e"].iloc[0],
            "share_within_2_laps": df_th["share_within_2_mode_e"].iloc[0]
        })
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(cfg["paths"]["reports_dir"], "cliff_sensitivity.csv")
    out_df.to_csv(out_path, index=False)
    print("Cliff Sensitivity Results:")
    print(out_df)
    return out_df
