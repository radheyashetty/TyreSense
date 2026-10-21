import os
import pandas as pd
from src.download import load_config
from src.fuel import fuel_correct, compute_fuel_coefficient


def is_wet_race(dataframe):
    """Check if race session used any intermediate or wet tyres."""
    compounds = dataframe["Compound"].dropna().str.upper()
    return compounds.isin(["INTERMEDIATE", "WET"]).any()


def filter_step(df, mask, filter_name, attrition_list):
    """Apply boolean mask to dataframe and record dropped rows for attrition."""
    rows_dropped = len(df) - int(mask.sum())
    filtered_df = df[mask].copy()
    attrition_list.append({"filter_name": filter_name, "rows_dropped": rows_dropped, "rows_flagged": 0, "rows_remaining": len(filtered_df)})
    return filtered_df


def apply_basic_filters(df, attrition_list):
    """Filter lap 1, pit laps, SC/VSC, deleted/inaccurate, and missing fields."""
    df = filter_step(df, df["LapNumber"] > 1, "drop_lap_one", attrition_list)
    pit_mask = df["PitInTime"].isna() & df["PitOutTime"].isna()
    df = filter_step(df, pit_mask, "drop_in_out_laps", attrition_list)
    sc_mask = ~df["TrackStatus"].astype(str).str.contains("[4567]")
    df = filter_step(df, sc_mask, "drop_sc_vsc", attrition_list)
    # Retain accurate laps: FastF1 boolean True/False; OpenF1 NaN defaults to True
    acc_mask = (df["IsAccurate"] != False)
    if "Deleted" in df.columns:
        acc_mask = acc_mask & (~(df["Deleted"] == True))
    df = filter_step(df, acc_mask, "drop_inaccurate_deleted", attrition_list)
    req_cols = ["LapTime", "TrackTemp", "TyreLife", "Compound"]
    valid_mask = df[req_cols].notna().all(axis=1) & (df["Compound"].astype(str) != "None")
    return filter_step(df, valid_mask, "drop_missing_fields", attrition_list)


def split_stints_and_gaps(dataframe, max_gap=3):
    """Split stints on gaps greater than max_gap and assign gap masks."""
    stint_dfs, stint_counter = [], 1
    for driver in dataframe["Driver"].unique():
        driver_df = dataframe[dataframe["Driver"] == driver].sort_values("LapNumber")
        for orig_stint in driver_df["Stint"].unique():
            sub = driver_df[driver_df["Stint"] == orig_stint].copy()
            if len(sub) == 0:
                continue
            sub["lap_diff"] = sub["LapNumber"].diff().fillna(1.0)
            sub["is_split"] = (sub["lap_diff"] > (max_gap + 1)).cumsum()
            for _, split_group in sub.groupby("is_split"):
                group = split_group.copy()
                group["stint_id"] = f"{group['race_id'].iloc[0]}_{driver}_{stint_counter}"
                stint_counter += 1
                group["gap_mask"] = (group["lap_diff"] > 1.0).astype(int)
                group["stint_lap"] = range(1, len(group) + 1)
                stint_dfs.append(group)
    return pd.concat(stint_dfs, ignore_index=True) if stint_dfs else pd.DataFrame()


def flag_spikes_offline(stint, threshold_s=2.0, recover_s=1.0, window=3):
    """Flag offline traffic/lockup spikes where pace recovers on next lap."""
    times = stint["lap_time_fc_s"].values
    n = len(times)
    is_spike = [False] * n
    for i in range(window, n - 1):
        prev = [times[j] for j in range(i - window, i) if not is_spike[j]]
        if prev:
            med = float(pd.Series(prev).median())
            if (times[i] - med > threshold_s) and (abs(times[i + 1] - med) <= recover_s):
                is_spike[i] = True
    return pd.Series(is_spike, index=stint.index)


def flag_spikes_causal(history, threshold_s=2.0, window=3):
    """Flag causal spike for current lap looking only backwards."""
    times = history["lap_time_fc_s"].values
    if len(times) <= window:
        return False
    med = float(pd.Series(times[-window - 1 : -1]).median())
    return bool(times[-1] - med > threshold_s)


def detect_stint_cliff(stint_laps, times, baseline_time_s, threshold_s=1.0):
    """Find first lap >= 4 where degradation >= threshold for 2 consecutive laps."""
    n = len(times)
    is_deg = (times >= baseline_time_s + threshold_s)
    for i in range(n - 1):
        if stint_laps[i] >= 4 and is_deg[i] and is_deg[i + 1]:
            return int(stint_laps[i]), False
    return int(stint_laps[-1]), True


def compute_stint_labels(clean_laps_df, threshold_s=1.0):
    """Compute early-laps baseline, usable life laps, and censoring flag."""
    stints_records = []
    for stint_id in clean_laps_df["stint_id"].unique():
        sub = clean_laps_df[clean_laps_df["stint_id"] == stint_id].sort_values("stint_lap")
        early_laps = sub[sub["stint_lap"] <= 3]
        baseline_time_s = float(early_laps["lap_time_fc_s"].mean())
        life_laps, censored = detect_stint_cliff(
            sub["stint_lap"].values, sub["lap_time_fc_s"].values, baseline_time_s, threshold_s
        )
        stints_records.append({
            "stint_id": stint_id, "baseline_time_s": baseline_time_s,
            "life_laps": life_laps, "censored": censored
        })
    return pd.DataFrame(stints_records)


def prepare_race_columns(raw_df, config):
    """Format and derive race columns including track-scaled fuel correction."""
    df = raw_df.copy()
    year = int(df["Year"].iloc[0])
    round_num = int(df["RoundNumber"].iloc[0])
    df["race_id"] = f"{year}_{round_num:02d}"
    df["track_id"] = df["EventName"]
    df["driver"] = df["Driver"]
    df["total_laps"] = int(df["LapNumber"].max())
    df["lap_number"] = df["LapNumber"].astype(int)
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    df["track_temp_c"] = df["TrackTemp"]
    df["track_status"] = df["TrackStatus"].astype(str)
    assert not df["TyreLife"].isna().any(), "TyreLife must not contain nulls"
    df["tyre_age"] = df["TyreLife"].astype(int)
    df["compound"] = df["Compound"]
    fuel_c = compute_fuel_coefficient(df["track_id"].iloc[0], config)
    df["lap_time_fc_s"] = fuel_correct(df["lap_time_s"], df["lap_number"], df["total_laps"], c=fuel_c)
    return df


def apply_spike_flagging(clean_laps, config, attrition):
    """Optionally flag pace spikes in clean laps if configured."""
    if not config.get("spike", {}).get("enabled", False) or clean_laps.empty:
        return clean_laps
    th, rec, win = config["spike"]["threshold_s"], config["spike"]["recover_s"], config["spike"]["window"]
    flags = [flag_spikes_offline(grp, th, rec, win) for _, grp in clean_laps.groupby("stint_id")]
    spike_flags = pd.concat(flags).reindex(clean_laps.index).fillna(False)
    clean_laps["gap_mask"] = (clean_laps["gap_mask"].values | spike_flags.astype(int).values)
    attrition.append({"filter_name": "flag_spikes", "rows_dropped": 0, "rows_flagged": int(spike_flags.sum()), "rows_remaining": len(clean_laps)})
    return clean_laps


def clean_race_dataframe(raw_df, config):
    """Execute complete cleaning pipeline for a single raw race dataframe."""
    initial_count = len(raw_df)
    attrition = [{"filter_name": "initial_rows", "rows_dropped": 0, "rows_flagged": 0, "rows_remaining": initial_count}]
    if is_wet_race(raw_df):
        attrition.append({"filter_name": "drop_wet_race", "rows_dropped": initial_count, "rows_flagged": 0, "rows_remaining": 0})
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(attrition)

    attrition.append({"filter_name": "drop_wet_race", "rows_dropped": 0, "rows_flagged": 0, "rows_remaining": initial_count})
    df = apply_basic_filters(raw_df, attrition)
    assert not df["TyreLife"].isna().any(), "Raw TyreLife contains nulls after basic filters"
    df = prepare_race_columns(df, config)
    split_df = split_stints_and_gaps(df, max_gap=config["cleaning"]["max_gap_laps"])
    stint_lengths = split_df.groupby("stint_id")["stint_lap"].transform("max")
    clean_laps = filter_step(split_df, stint_lengths >= config["cleaning"]["min_stint_laps"], "drop_short_stints", attrition)
    clean_laps = apply_spike_flagging(clean_laps, config, attrition)

    cols = ["race_id", "track_id", "driver", "stint_id", "stint_lap", "tyre_age", "compound",
            "lap_number", "total_laps", "lap_time_s", "lap_time_fc_s", "track_temp_c", "track_status", "gap_mask"]
    clean_laps = clean_laps[cols].copy()
    stints_df = compute_stint_labels(clean_laps, config["cleaning"]["usable_life_threshold_s"])
    return clean_laps, stints_df, pd.DataFrame(attrition)


def process_and_save_all(config_path="config.yaml"):
    """Clean all raw parquet files and save processed datasets with attrition report."""
    config = load_config(config_path)
    raw_dir, proc_dir, rep_dir = config["paths"]["raw_dir"], config["paths"]["processed_dir"], config["paths"]["reports_dir"]
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    all_laps, all_stints, all_att = [], [], []
    for file_name in sorted(os.listdir(raw_dir)):
        if file_name.endswith(".parquet"):
            raw = pd.read_parquet(os.path.join(raw_dir, file_name))
            laps, stints, att = clean_race_dataframe(raw, config)
            if not laps.empty:
                all_laps.append(laps)
                all_stints.append(stints)
            all_att.append(att)
    if all_laps:
        pd.concat(all_laps, ignore_index=True).to_parquet(os.path.join(proc_dir, "laps.parquet"), index=False)
        pd.concat(all_stints, ignore_index=True).to_parquet(os.path.join(proc_dir, "stints.parquet"), index=False)
        att_out = pd.concat(all_att, ignore_index=True).groupby("filter_name", sort=False).sum().reset_index()
        att_out.to_csv(os.path.join(rep_dir, "attrition.csv"), index=False)
