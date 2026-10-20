import os
import glob
import pandas as pd

RAW_PARQUETS = sorted(glob.glob("data/raw/*.parquet"))
VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def get_raw_parquets():
    """Return sorted list of all valid raw parquets."""
    return sorted(glob.glob("data/raw/*.parquet"))


def check_schema_and_dtypes(df):
    """Validate 39-column schema."""
    return len(df.columns) == 39


def check_duplicates(df):
    """Check duplicate Driver and LapNumber rows."""
    return int(df.duplicated(subset=["Driver", "LapNumber"]).sum()) == 0


def check_lap_start_at_one(df):
    """Check that each driver starts at LapNumber 1."""
    for drv, g in df.groupby("Driver"):
        laps = sorted(g["LapNumber"].dropna().tolist())
        if not laps or laps[0] != 1.0:
            return False, f"{drv} starts at {laps[0] if laps else None}"
    return True, ""


def check_contiguous_laps(df):
    """Check completeness rule: <= 5% missing laps and < 4 consecutive missing."""
    tot_missing = 0
    tot_expected = 0
    for drv, g in df.groupby("Driver"):
        laps = set(g["LapNumber"].dropna().astype(int))
        if laps:
            max_l = max(laps)
            exp = set(range(1, max_l + 1))
            missing = exp - laps
            tot_missing += len(missing)
            tot_expected += max_l
            consec = 0
            for l in range(1, max_l + 1):
                if l not in laps:
                    consec += 1
                    if consec >= 4:
                        return False, f"{drv} has >=4 consecutive missing laps"
                else:
                    consec = 0
    share = (tot_missing / tot_expected) if tot_expected > 0 else 0.0
    if share > 0.05:
        return False, f"missing share {share:.2%} > 5%"
    return True, ""


def check_lap_times_valid(df):
    """Check lap times between 50s and 400s on green/flying laps."""
    for _, r in df.iterrows():
        lt = r["LapTime"]
        ts = str(r.get("TrackStatus", "1"))
        is_pit = pd.notna(r.get("PitInTime")) or pd.notna(r.get("PitOutTime"))
        if pd.notna(lt) and not is_pit and ts == "1":
            sec = lt.total_seconds()
            if sec < 50.0 or sec > 400.0:
                return False, f"green lap {r['LapNumber']} time {sec:.1f}s out of bounds"
    return True, ""


def check_compounds_valid(df):
    """Check Compound is in the 5 valid values (or None string from FastF1)."""
    comps = set(df["Compound"].dropna().unique())
    invalid = comps - VALID_COMPOUNDS
    if invalid:
        return False, f"Invalid compounds: {invalid}"
    return True, ""


def check_tyre_life_rules(df):
    """Check TyreLife >= 1 and no reset without pit or compound change."""
    tls = df["TyreLife"].dropna()
    if (tls < 1).any():
        return False, f"TyreLife min is {tls.min()}"
    for drv, g in df.groupby("Driver"):
        g = g.sort_values("LapNumber")
        prev_tl, prev_c, prev_s = None, None, None
        for _, r in g.iterrows():
            tl, c, s = r["TyreLife"], r["Compound"], r["Stint"]
            is_out = pd.notna(r["PitOutTime"])
            if prev_tl is not None and pd.notna(tl):
                if tl < prev_tl and not is_out and c == prev_c and s == prev_s:
                    return False, f"{drv} lap {r['LapNumber']} TL reset without pit/comp"
            prev_tl, prev_c, prev_s = tl, c, s
    return True, ""


def check_weather_bounds(df):
    """Check TrackTemp 10-70, AirTemp 5-50, Humidity 0-100."""
    tt, at, hum = df["TrackTemp"].dropna(), df["AirTemp"].dropna(), df["Humidity"].dropna()
    if len(tt) > 0 and ((tt < 10) | (tt > 70)).any():
        return False, f"TrackTemp out of bounds: [{tt.min()}, {tt.max()}]"
    if len(at) > 0 and ((at < 5) | (at > 50)).any():
        return False, f"AirTemp out of bounds: [{at.min()}, {at.max()}]"
    if len(hum) > 0 and ((hum < 0) | (hum > 100)).any():
        return False, f"Humidity out of bounds: [{hum.min()}, {hum.max()}]"
    return True, ""
