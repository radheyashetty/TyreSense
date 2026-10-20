import re
import pandas as pd
import numpy as np
from src.validate_raw import get_part1_columns


def parse_deleted_laps(rc_data):
    """Parse car number, lap, and reason for deleted lap times from race control."""
    deleted_map = {}
    for m in rc_data:
        text = str(m.get("message", ""))
        if "DELETED" in text.upper():
            car_m = re.search(r"CAR\s+(\d+)", text, re.IGNORECASE)
            lap_m = re.search(r"LAP\s+(\d+)", text, re.IGNORECASE)
            if car_m and lap_m:
                car = int(car_m.group(1))
                lap = int(lap_m.group(1))
                deleted_map[(car, lap)] = text
    return deleted_map


def parse_sc_intervals(rc_data):
    """Extract disruption time intervals from race control messages."""
    intervals, current = [], None
    for m in rc_data:
        msg = str(m.get("message", "")).upper()
        t = pd.to_datetime(m.get("date"), format="ISO8601")
        if "VIRTUAL SAFETY CAR DEPLOYED" in msg:
            current = {"type": "6", "start": t, "end": None}
        elif "SAFETY CAR DEPLOYED" in msg:
            current = {"type": "4", "start": t, "end": None}
        elif ("VIRTUAL SAFETY CAR ENDING" in msg or "SAFETY CAR IN THIS LAP" in msg or "TRACK CLEAR" in msg) and current:
            current["end"] = t
            intervals.append(current)
            current = None
    if current and current["end"] is None:
        current["end"] = pd.Timestamp.max.tz_localize("UTC")
        intervals.append(current)
    return intervals


def compute_lap_track_status(t_start, t_end, intervals):
    """Determine FastF1-style TrackStatus string based on interval overlap."""
    if pd.isna(t_start) or pd.isna(t_end):
        return "1"
    for p in intervals:
        if t_start < p["end"] and t_end > p["start"]:
            return p["type"]
    return "1"


def get_stint_number(stint):
    """Return stint number as integer for sorting."""
    val = stint.get("stint_number")
    return int(val) if val is not None else 1


def build_stint_lookup(stints_data):
    """Map (driver_number, lap_number) to stint, compound, and tyre life."""
    lookup, driver_stints = {}, {}
    for row in stints_data:
        d = int(row.get("driver_number") or 0)
        driver_stints.setdefault(d, []).append(row)
    for d, s_list in driver_stints.items():
        s_list.sort(key=get_stint_number)
        for i, row in enumerate(s_list):
            s_v = row.get("stint_number")
            s_num = float(s_v) if s_v is not None else 1.0
            comp = str(row.get("compound") or "UNKNOWN").upper()
            age_start = int(row.get("tyre_age_at_start") or 0)
            l_start = int(row.get("lap_start") or 1)
            if i + 1 < len(s_list):
                nxt = s_list[i + 1].get("lap_start")
                l_end = (int(nxt) - 1) if nxt is not None else l_start
            else:
                raw_end = row.get("lap_end")
                end_base = int(raw_end) if raw_end is not None else l_start
                l_end = max(end_base, l_start) + 20
            l_end = max(l_end, l_start)
            for lap in range(l_start, l_end + 1):
                tl = float(age_start + (lap - l_start) + 1)
                lookup[(d, lap)] = (s_num, comp, tl)
    return lookup


def merge_weather_data(df, weather_data):
    """Merge nearest weather readings based on lap end timestamp."""
    w_df = pd.DataFrame(weather_data)
    if w_df.empty or "date" not in w_df.columns:
        df["TrackTemp"], df["AirTemp"], df["Humidity"], df["Rainfall"] = np.nan, np.nan, np.nan, False
        return df
    w_df["date"] = pd.to_datetime(w_df["date"], format="ISO8601")
    w_df = w_df.dropna(subset=["date"]).sort_values("date")
    df_sorted = df.sort_values("t_end")
    merged = pd.merge_asof(df_sorted, w_df, left_on="t_end", right_on="date", direction="nearest")
    merged = merged.sort_index()
    df["TrackTemp"] = merged["track_temperature"].astype(float).values
    df["AirTemp"] = merged["air_temperature"].astype(float).values
    df["Humidity"] = merged["humidity"].astype(float).values
    df["Rainfall"] = (merged["rainfall"].fillna(0) > 0).astype(bool).values
    return df


def assign_timing_and_driver_fields(df, drivers):
    """Assign driver identifiers, timestamps, and lap elapsed times."""
    acronym_map = {d["driver_number"]: d.get("name_acronym", str(d["driver_number"])) for d in drivers.values()}
    df["Driver"] = df["driver_number"].map(acronym_map)
    df["DriverNumber"] = df["driver_number"].astype(str)
    df["LapNumber"] = df["lap_number"].astype(float)
    df["LapTime"] = pd.to_timedelta(df["lap_duration"], unit="s")
    df["t_start"] = pd.to_datetime(df["date_start"], format="ISO8601")
    t0 = df["t_start"].dropna().min()
    df["t_start"] = df["t_start"].fillna(t0)
    df["t_end"] = df["t_start"] + pd.to_timedelta(df["lap_duration"].fillna(0), unit="s")
    df["LapStartTime"] = df["t_start"] - t0
    df["Time"] = df["t_end"] - t0
    df["LapStartDate"] = df["t_start"].dt.tz_localize(None)


def assign_stints_and_sectors(df, bundle, drivers):
    """Assign tyre stint metadata, team names, sector times, and trap speeds."""
    stints_map = build_stint_lookup(bundle.get("stints", []))
    stints, comps, tls = [], [], []
    for _, row in df.iterrows():
        key = (int(row["driver_number"]), int(row["lap_number"]))
        s_val = stints_map.get(key, (1.0, "UNKNOWN", float(row["lap_number"])))
        stints.append(s_val[0])
        comps.append(s_val[1])
        tls.append(s_val[2])
    df["Stint"], df["Compound"], df["TyreLife"] = stints, comps, tls
    team_map = {d["driver_number"]: d.get("team_name", "") for d in drivers.values()}
    df["Team"] = df["driver_number"].map(team_map)
    df["Sector1Time"] = pd.to_timedelta(df["duration_sector_1"], unit="s") if "duration_sector_1" in df.columns else pd.NaT
    df["Sector2Time"] = pd.to_timedelta(df["duration_sector_2"], unit="s") if "duration_sector_2" in df.columns else pd.NaT
    df["Sector3Time"] = pd.to_timedelta(df["duration_sector_3"], unit="s") if "duration_sector_3" in df.columns else pd.NaT
    df["SpeedI1"] = df["i1_speed"].astype(float) if "i1_speed" in df.columns else np.nan
    df["SpeedI2"] = df["i2_speed"].astype(float) if "i2_speed" in df.columns else np.nan
    df["SpeedFL"] = np.nan
    df["SpeedST"] = df["st_speed"].astype(float) if "st_speed" in df.columns else np.nan


def assign_pit_and_race_control(df, bundle):
    """Assign pit in/out timedeltas, deleted laps, and track status."""
    df["PitOutTime"] = pd.to_timedelta([np.nan] * len(df))
    df["PitInTime"] = pd.to_timedelta([np.nan] * len(df))
    if "is_pit_out_lap" not in df.columns:
        df["is_pit_out_lap"] = False
    for _, grp in df.groupby("DriverNumber"):
        grp_sorted = grp.sort_values("LapNumber")
        out_indices = grp_sorted[grp_sorted["is_pit_out_lap"] == True].index
        for idx in out_indices:
            df.loc[idx, "PitOutTime"] = df.loc[idx, "LapStartTime"]
            pos = grp_sorted.index.get_loc(idx)
            if pos > 0:
                prev_idx = grp_sorted.index[pos - 1]
                df.loc[prev_idx, "PitInTime"] = df.loc[prev_idx, "Time"]
    del_map = parse_deleted_laps(bundle.get("race_control", []))
    d_flags, d_reasons = [], []
    for _, row in df.iterrows():
        d_key = (int(row["driver_number"]), int(row["lap_number"]))
        is_del = bool(d_key in del_map)
        d_flags.append(is_del)
        d_reasons.append(del_map[d_key] if is_del else None)
    df["Deleted"], df["DeletedReason"] = d_flags, d_reasons
    sc_intervals = parse_sc_intervals(bundle.get("race_control", []))
    status_list = [compute_lap_track_status(s, e, sc_intervals) for s, e in zip(df["t_start"], df["t_end"])]
    df["TrackStatus"] = status_list


def assemble_race_dataframe(df, event_info, round_number, year):
    """Fill metadata and reorder columns to match the 39-column Part 1 schema."""
    df["EventName"] = str(event_info.get("event_name", event_info.get("circuit_short_name", "Grand Prix")))
    df["Location"] = str(event_info.get("location", event_info.get("circuit_short_name", "Circuit")))
    df["RoundNumber"], df["Year"] = int(round_number), int(year)
    df["Position"], df["IsPersonalBest"] = np.nan, None
    df["FreshTyre"], df["FastF1Generated"], df["IsAccurate"] = False, False, np.nan
    cols = get_part1_columns()
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols]


def convert_openf1_bundle(bundle, event_info, round_number, year):
    """Convert raw OpenF1 bundle dictionaries into standard 39-column DataFrame."""
    laps_data = bundle.get("laps", [])
    if not laps_data:
        return pd.DataFrame(columns=get_part1_columns())
    df = pd.DataFrame(laps_data)
    if "driver_number" in df.columns and "lap_number" in df.columns:
        df = df.drop_duplicates(subset=["driver_number", "lap_number"]).reset_index(drop=True)
    drivers = {d["driver_number"]: d for d in bundle.get("drivers", [])}
    assign_timing_and_driver_fields(df, drivers)
    assign_stints_and_sectors(df, bundle, drivers)
    assign_pit_and_race_control(df, bundle)
    df = merge_weather_data(df, bundle.get("weather", []))
    return assemble_race_dataframe(df, event_info, round_number, year)
