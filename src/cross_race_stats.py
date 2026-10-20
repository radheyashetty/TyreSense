import os
import glob
import pandas as pd
import numpy as np


def load_all_parquets_summary():
    """Load summary statistics for all 91 valid races."""
    parquets = sorted(glob.glob("data/raw/*.parquet"))
    rows = []
    for p in parquets:
        rid = os.path.basename(p).replace(".parquet", "")
        parts = rid.split("_")
        season = int(parts[1])
        round_num = int(parts[2])
        df = pd.read_parquet(p)
        ev_name = str(df["EventName"].iloc[0]) if len(df) > 0 and "EventName" in df.columns else "Unknown"
        lts = df["LapTime"].dropna()
        sec = [t.total_seconds() for t in lts if t.total_seconds() > 0]
        med_lt = float(np.median(sec)) if sec else np.nan
        mean_lt = float(np.mean(sec)) if sec else np.nan
        tls = df["TyreLife"].dropna().tolist()
        med_tl = float(np.median(tls)) if tls else np.nan
        ts_str = df["TrackStatus"].astype(str)
        sc_share = float((ts_str.str.contains("4") | ts_str.str.contains("6")).mean()) if len(df) > 0 else 0.0
        del_count = int((df["Deleted"] == True).sum()) if "Deleted" in df.columns else 0
        src = "fastf1" if season == 2022 else "openf1"
        rows.append({
            "race_id": rid, "season": season, "round": round_num, "event_name": ev_name,
            "source": src, "laps": len(df), "median_laptime": med_lt, "mean_laptime": mean_lt,
            "median_tyrelife": med_tl, "sc_vsc_share": sc_share, "deleted_laps": del_count
        })
    return pd.DataFrame(rows)


def analyze_circuit_shifts(df_summary):
    """Find circuit-seasons where median lap time deviates > 15% from other seasons."""
    shifts = []
    for ev, g in df_summary.groupby("event_name"):
        if len(g) <= 1:
            continue
        all_meds = g["median_laptime"].dropna().tolist()
        for _, r in g.iterrows():
            other_meds = [m for m in all_meds if m != r["median_laptime"]]
            if not other_meds:
                continue
            bench = float(np.median(other_meds))
            diff_pct = abs(r["median_laptime"] - bench) / bench * 100.0
            if diff_pct > 15.0:
                shifts.append({
                    "event_name": ev, "season": r["season"], "source": r["source"],
                    "season_median": round(r["median_laptime"], 3),
                    "benchmark_median": round(bench, 3), "diff_pct": round(diff_pct, 2)
                })
    return pd.DataFrame(shifts)


def compute_source_level_shifts():
    """Compute overall distribution differences between FastF1 (2022) and OpenF1 (2023)."""
    p22 = sorted(glob.glob("data/raw/race_2022_*.parquet"))
    p23 = sorted(glob.glob("data/raw/race_2023_*.parquet"))
    df22 = pd.concat([pd.read_parquet(p) for p in p22], ignore_index=True)
    df23 = pd.concat([pd.read_parquet(p) for p in p23], ignore_index=True)
    tl22 = df22["TyreLife"].dropna()
    tl23 = df23["TyreLife"].dropna()
    cm22 = (df22["Compound"].value_counts(normalize=True) * 100).round(2).to_dict()
    cm23 = (df23["Compound"].value_counts(normalize=True) * 100).round(2).to_dict()
    return {
        "tyrelife_quantiles_2022": [round(float(q), 1) for q in np.percentile(tl22, [10, 25, 50, 75, 90])],
        "tyrelife_quantiles_2023": [round(float(q), 1) for q in np.percentile(tl23, [10, 25, 50, 75, 90])],
        "compound_mix_2022": cm22,
        "compound_mix_2023": cm23
    }


def run_cross_race_analysis():
    """Execute analysis and write reports/testing/cross_race_stats.csv."""
    os.makedirs("reports/testing", exist_ok=True)
    df_sum = load_all_parquets_summary()
    df_sum.to_csv("reports/testing/cross_race_stats.csv", index=False)
    shifts = analyze_circuit_shifts(df_sum)
    source_shifts = compute_source_level_shifts()
    print("=== CROSS-RACE STATISTICS SUMMARY ===")
    print(f"Total races analyzed: {len(df_sum)}")
    print(f"Circuit-seasons with >15% median lap time shift: {len(shifts)}")
    if len(shifts) > 0:
        print(shifts.to_string(index=False))
    print("\nSource level comparison (2022 FastF1 vs 2023 OpenF1):")
    print("TyreLife Q10, Q25, Q50, Q75, Q90:")
    print("  2022:", source_shifts["tyrelife_quantiles_2022"])
    print("  2023:", source_shifts["tyrelife_quantiles_2023"])
    print("Compound mix (%):")
    print("  2022:", source_shifts["compound_mix_2022"])
    print("  2023:", source_shifts["compound_mix_2023"])


if __name__ == "__main__":
    run_cross_race_analysis()
