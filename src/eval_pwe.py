import os, numpy as np, pandas as pd
from src.features import fit_baseline_curves, eval_curve
from src.claim_eval import forecast_stint_pace
from src.live import compute_cliff_life, update_k_factor
from src.clean import flag_spikes_causal


def get_stint_sub(val_laps, stint_id):
    """Retrieve sorted laps for a stint."""
    return val_laps[val_laps["stint_id"] == stint_id].sort_values("stint_lap")


def compute_online_k(sub, curves, g_curve, lap):
    """Compute online wear multiplier k using clean baseline step increments."""
    k, k_hist = 1.0, []
    track, cmp_name = sub["track_id"].iloc[0], sub["compound"].iloc[0]
    params = curves.get((track, cmp_name), g_curve)
    raw_curve = eval_curve(params, sub["tyre_age"].values)
    for i in range(1, lap):
        s_lap = int(sub["stint_lap"].iloc[i])
        act_t = float(sub["lap_time_fc_s"].iloc[i])
        prev_act = float(sub["lap_time_fc_s"].iloc[i - 1])
        delta_b = float(raw_curve[i] - raw_curve[i - 1])
        spike = flag_spikes_causal(sub.iloc[: i + 1])
        if s_lap >= 3:
            k, k_hist = update_k_factor(k, k_hist, act_t, prev_act, delta_b, spike)
    return k


def get_predictions_at_lap(sub, curves, g_curve, lap, meds):
    """Compute absolute (capped at 22) and relative forecasts across modes."""
    f_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=1)
    r_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=lap - 1)
    f_c = compute_cliff_life(f_deg, 1.0)
    r_c = compute_cliff_life(r_deg, 1.0)

    k = compute_online_k(sub, curves, g_curve, lap)
    e_c = compute_cliff_life(k * r_deg, 1.0)

    l_f = min(22.0, float(2 + f_c)) if f_c is not None else 22.0
    l_r = min(22.0, float(lap + r_c)) if r_c is not None else 22.0
    l_e = min(22.0, float(lap + e_c)) if e_c is not None else 22.0
    cmp_name = sub["compound"].iloc[0]
    l_m = min(22.0, float(meds.get(cmp_name, 12.0)))

    r_f = min(20.0, max(0.0, float((2 + f_c) - lap))) if f_c is not None else 20.0
    r_r = min(20.0, float(r_c)) if r_c is not None else 20.0
    r_e = min(20.0, float(e_c)) if e_c is not None else 20.0
    r_m = min(20.0, max(0.0, float(meds.get(cmp_name, 12.0) - lap)))

    return (l_f, l_r, l_e, 22.0, l_m), (r_f, r_r, r_e, 20.0, r_m), f_c is not None


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


def eval_uncensored_stints(alive, val_laps, curves, g_curve, lap, meds):
    """Evaluate absolute and relative PWE across modes on alive-at-t cohort."""
    p_abs = {m: [] for m in ["f", "r", "e", "22", "m"]}
    p_rel = {m: [] for m in ["f", "r", "e", "20", "m"]}
    r_diffs_rf, r_diffs_er = {}, {}
    froz_cross = 0

    for _, s in alive.iterrows():
        sub = get_stint_sub(val_laps, s["stint_id"])
        (lf, lr, le, l22, lm), (rf, rr, re, r20, rm), crossed = get_predictions_at_lap(sub, curves, g_curve, lap, meds)
        if crossed:
            froz_cross += 1
        l_true = min(22.0, float(s["life_laps"]))
        r_true = min(20.0, max(0.0, float(s["life_laps"] - lap)))

        for k, v in zip(["f", "r", "e", "22", "m"], [lf, lr, le, l22, lm]):
            p_abs[k].append(abs(v - l_true))
        for k, v in zip(["f", "r", "e", "20", "m"], [rf, rr, re, r20, rm]):
            p_rel[k].append(abs(v - r_true))

        rid = sub["race_id"].iloc[0]
        r_diffs_rf.setdefault(rid, []).append(abs(lf - l_true) - abs(lr - l_true))
        r_diffs_er.setdefault(rid, []).append(abs(lr - l_true) - abs(le - l_true))

    return p_abs, p_rel, froz_cross, r_diffs_rf, r_diffs_er


def eval_censored_stints(censored, val_laps, curves, g_curve, lap):
    """Compute survival-consistency rate on censored stints with common horizon."""
    c_f, c_r, c_22 = [], [], []
    for _, s in censored.iterrows():
        sub = get_stint_sub(val_laps, s["stint_id"])
        st_len = float(len(sub))
        f_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=1)
        r_deg = forecast_stint_pace(sub, curves, g_curve, lap_idx=lap - 1)
        fc = compute_cliff_life(f_deg, 1.0)
        rc = compute_cliff_life(r_deg, 1.0)
        lf = min(22.0, float(2 + fc)) if fc is not None else 22.0
        lr = min(22.0, float(lap + rc)) if rc is not None else 22.0
        c_f.append(lf >= min(22.0, st_len))
        c_r.append(lr >= min(22.0, st_len))
        c_22.append(22.0 >= min(22.0, st_len))
    return len(c_r), float(np.mean(c_f)), float(np.mean(c_r)), float(np.mean(c_22))


def format_fair_record(lap, alive, excl, pa, pr, r_diffs_rf, r_diffs_er, cross_count, c_res):
    """Format dictionary containing fair evaluation metrics for a single lap."""
    m_rf, ci_s_rf = run_stint_boot(np.array(pa["f"]) - np.array(pa["r"]))
    _, ci_c_rf = run_cluster_boot(r_diffs_rf)
    m_er, ci_s_er = run_stint_boot(np.array(pa["r"]) - np.array(pa["e"]))
    _, ci_c_er = run_cluster_boot(r_diffs_er)
    race_means_rf = [float(np.mean(r_diffs_rf[r])) for r in r_diffs_rf]
    var_between = float(np.var(race_means_rf, ddof=1)) if len(race_means_rf) > 1 else 0.0
    c_n, c_f, c_r, c_22 = c_res
    return {
        "stint_lap": lap, "alive_n": len(alive), "excluded_observed": excl,
        "n_races": len(r_diffs_rf), "froz_crossing_share": round(float(cross_count / len(alive)), 3),
        "pwe_abs_frozen": round(float(np.mean(pa["f"])), 2), "pwe_abs_rolling": round(float(np.mean(pa["r"])), 2),
        "pwe_abs_mode_e": round(float(np.mean(pa["e"])), 2), "pwe_abs_always22": round(float(np.mean(pa["22"])), 2),
        "pwe_abs_median": round(float(np.mean(pa["m"])), 2),
        "pwe_rel_frozen": round(float(np.mean(pr["f"])), 2), "pwe_rel_rolling": round(float(np.mean(pr["r"])), 2),
        "pwe_rel_mode_e": round(float(np.mean(pr["e"])), 2), "pwe_rel_always20": round(float(np.mean(pr["20"])), 2),
        "pwe_rel_median": round(float(np.mean(pr["m"])), 2),
        "rolling_adv_abs": round(m_rf, 2), "ci_95_stint_rf": f"[{ci_s_rf[0]:.2f}, {ci_s_rf[1]:.2f}]",
        "ci_95_cluster_rf": f"[{ci_c_rf[0]:.2f}, {ci_c_rf[1]:.2f}]", "between_race_var_rf": round(var_between, 4),
        "mode_e_adv_abs": round(m_er, 2), "ci_95_stint_er": f"[{ci_s_er[0]:.2f}, {ci_s_er[1]:.2f}]",
        "ci_95_cluster_er": f"[{ci_c_er[0]:.2f}, {ci_c_er[1]:.2f}]",
        "censored_n": c_n, "cens_surv_frozen": round(c_f, 3), "cens_surv_rolling": round(c_r, 3),
        "cens_surv_always22": round(c_22, 3)
    }


def compute_lap_record(lap, uncensored, censored, val_laps, curves, g_curve, meds):
    """Generate fair comparison record for a single evaluation lap."""
    alive = uncensored[(uncensored["len"] >= lap) & (uncensored["life_laps"] >= lap)]
    excl = len(uncensored[uncensored["len"] >= lap]) - len(alive)
    pa, pr, cross_count, r_diffs_rf, r_diffs_er = eval_uncensored_stints(alive, val_laps, curves, g_curve, lap, meds)
    c_sub = censored[censored["len"] >= lap]
    c_res = eval_censored_stints(c_sub, val_laps, curves, g_curve, lap)
    return format_fair_record(lap, alive, excl, pa, pr, r_diffs_rf, r_diffs_er, cross_count, c_res)


def main():
    """Run fair EXP-9 evaluation across laps 3, 5, 7, 10 and save reports."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    stints = pd.read_parquet("data/processed/stints.parquet")
    curves, g_curve = fit_baseline_curves(tr_laps)

    sc = tr_laps[["stint_id", "compound"]].drop_duplicates()
    ts = stints.merge(sc, on="stint_id")
    meds = ts[ts["stint_id"].isin(tr_laps["stint_id"]) & ~ts["censored"]].groupby("compound")["life_laps"].median().to_dict()

    counts = va_laps.groupby("stint_id").size()
    stints["len"] = stints["stint_id"].map(counts)
    va_stints = stints[stints["stint_id"].isin(va_laps["stint_id"].unique())]
    uncensored = va_stints[~va_stints["censored"]]
    censored = va_stints[va_stints["censored"]]

    records = [compute_lap_record(l, uncensored, censored, va_laps, curves, g_curve, meds) for l in [3, 5, 7, 10]]
    df = pd.DataFrame(records)
    os.makedirs("reports/overnight", exist_ok=True)
    df.to_csv("reports/overnight/exp9_fair.csv", index=False)
    df.to_csv("reports/overnight/exp9_official.csv", index=False)
    print("Fair EXP-9 completed! Results saved to reports/overnight/exp9_fair.csv and exp9_official.csv")
    print(df.to_string())


if __name__ == "__main__":
    main()
