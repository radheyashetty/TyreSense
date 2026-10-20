import os
import pandas as pd
from src.openf1_source import fetch_race_bundle
from src.openf1_convert import convert_openf1_bundle


def compute_cross_metrics(merged):
    """Compute agreement metrics between FastF1 and OpenF1 for flying and lap 1."""
    flying = merged[merged["LapNumber"] > 1]
    f1_fly = flying["LapTime_f1"].dt.total_seconds()
    of1_fly = flying["LapTime_of1"].dt.total_seconds()
    val_fly = f1_fly.notna() & of1_fly.notna()
    lt_fly = float((abs(f1_fly[val_fly] - of1_fly[val_fly]) <= 0.005).mean()) if val_fly.sum() > 0 else 0.0
    lap1 = merged[merged["LapNumber"] == 1]
    f1_l1 = lap1["LapTime_f1"].dt.total_seconds()
    of1_l1 = lap1["LapTime_of1"].dt.total_seconds()
    val_l1 = f1_l1.notna() & of1_l1.notna()
    lt_l1 = float((abs(f1_l1[val_l1] - of1_l1[val_l1]) <= 0.005).mean()) if val_l1.sum() > 0 else 0.0
    tl_m = float((merged["TyreLife_f1"] == merged["TyreLife_of1"]).mean())
    cp_m = float((merged["Compound_f1"].str.upper() == merged["Compound_of1"].str.upper()).mean())
    f1_in, of1_in = merged["PitInTime_f1"].notna(), merged["PitInTime_of1"].notna()
    f1_out, of1_out = merged["PitOutTime_f1"].notna(), merged["PitOutTime_of1"].notna()
    pt_m = float(((f1_in == of1_in) & (f1_out == of1_out)).mean())
    f1_sc = merged["TrackStatus_f1"].astype(str).str.contains("[4567]")
    of1_sc = merged["TrackStatus_of1"].astype(str).str.contains("[4567]")
    sc_m = float((f1_sc == of1_sc).mean())
    return lt_fly, lt_l1, tl_m, cp_m, pt_m, sc_m


def find_differing_stints(merged, race_id):
    """Find and format all stints where TyreLife differs between FastF1 and OpenF1."""
    diff_laps = merged[merged["TyreLife_f1"] != merged["TyreLife_of1"]]
    differing = []
    for (drv, st), g in diff_laps.groupby(["Driver", "Stint_f1"]):
        diff_val = int(g["TyreLife_of1"].iloc[0] - g["TyreLife_f1"].iloc[0])
        differing.append({"race_id": race_id, "driver": drv, "stint": float(st), "diff_laps": diff_val, "laps": len(g)})
    return differing


def compare_single_race(sk, year, round_num, event_name, location, f1_path):
    """Compare converted OpenF1 race against FastF1 parquet per (Driver, LapNumber)."""
    bundle = fetch_race_bundle(sk, cache_dir="data/openf1_cache")
    of1_df = convert_openf1_bundle(bundle, {"event_name": event_name, "location": location}, round_num, year)
    f1_df = pd.read_parquet(f1_path)
    merged = pd.merge(f1_df, of1_df, on=["Driver", "LapNumber"], suffixes=("_f1", "_of1"))
    lt_fly, lt_l1, tl_m, cp_m, pt_m, sc_m = compute_cross_metrics(merged)
    rid = f"{year}_{round_num:02d}"
    diff_stints = find_differing_stints(merged, rid)
    rec = {
        "race_id": rid, "laps_compared": len(merged),
        "lap_time_flying_5ms_share": round(lt_fly, 4), "lap_time_lap1_5ms_share": round(lt_l1, 4),
        "tyre_life_match_share": round(tl_m, 4), "compound_match_share": round(cp_m, 4),
        "pit_flags_match_share": round(pt_m, 4), "sc_vsc_match_share": round(sc_m, 4)
    }
    return rec, diff_stints


def run_cross_source_check(reports_dir="reports"):
    """Execute cross-source validation across 2023 R1 and R2 and save report."""
    os.makedirs(reports_dir, exist_ok=True)
    races = [
        (7953, 2023, 1, "Bahrain Grand Prix", "Sakhir", "data/raw_old_no_messages/race_2023_01.parquet"),
        (7779, 2023, 2, "Saudi Arabian Grand Prix", "Jeddah", "data/raw_old_no_messages/race_2023_02.parquet")
    ]
    records, all_stints = [], []
    for sk, year, r_num, ev_name, loc, f1_path in races:
        rec, diffs = compare_single_race(sk, year, r_num, ev_name, loc, f1_path)
        records.append(rec)
        all_stints.extend(diffs)
    df = pd.DataFrame(records)
    csv_path = os.path.join(reports_dir, "cross_source_check.csv")
    df.to_csv(csv_path, index=False)
    return df, all_stints


def print_cross_check_summary(df, all_stints):
    """Print formatted cross-check table with PASS/FAIL status and differing stints."""
    print("=== CROSS-SOURCE VALIDATION (OpenF1 vs FastF1) ===")
    print(df.to_string(index=False))
    print("\nEvaluation against amended thresholds:")
    tot_laps = df["laps_compared"].sum()
    tot_tl_match = sum(r["laps_compared"] * r["tyre_life_match_share"] for _, r in df.iterrows())
    overall_tl = tot_tl_match / tot_laps
    print(f"Overall TyreLife match share across both races: {overall_tl:.4f} (Threshold >= 95%: {'PASS' if overall_tl >= 0.95 else 'FAIL'})")
    for _, r in df.iterrows():
        rid = r["race_id"]
        lt_pass = "PASS" if r["lap_time_flying_5ms_share"] >= 0.99 else "FAIL"
        cp_pass = "PASS" if r["compound_match_share"] >= 0.99 else "FAIL"
        sc_pass = "PASS" if r["sc_vsc_match_share"] >= 0.98 else "FAIL"
        print(f"  {rid} LapTime (LapNumber>1 >=99%): {r['lap_time_flying_5ms_share']:.4f} -> {lt_pass} (Lap 1 info: {r['lap_time_lap1_5ms_share']:.4f})")
        print(f"  {rid} Compound (>=99%): {r['compound_match_share']:.4f} -> {cp_pass}")
        print(f"  {rid} SC/VSC (>=98%): {r['sc_vsc_match_share']:.4f} -> {sc_pass}")
        print(f"  {rid} Pit Flags: {r['pit_flags_match_share']:.4f}")
    print("\nDiffering TyreLife Stints across both races:")
    for s in all_stints:
        print(f"  {s['race_id']} Driver {s['driver']}, Stint {int(s['stint'])}: difference = {s['diff_laps']} laps ({s['laps']} laps in stint)")


def main():
    """CLI entry point for running cross source validation."""
    df, all_stints = run_cross_source_check()
    print_cross_check_summary(df, all_stints)


if __name__ == "__main__":
    main()
