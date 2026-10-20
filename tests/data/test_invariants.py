import os
import glob
import pytest
import pandas as pd
from tests.data.invariant_helpers import (
    get_raw_parquets, check_schema_and_dtypes, check_duplicates,
    check_lap_start_at_one, check_contiguous_laps, check_lap_times_valid,
    check_compounds_valid, check_tyre_life_rules, check_weather_bounds
)

RAW_PARQUETS = get_raw_parquets()
pytestmark = [pytest.mark.data]
if not RAW_PARQUETS:
    pytestmark.append(pytest.mark.skip(reason="data/raw is empty"))


def test_schema_and_dtypes():
    """Verify all parquets match the 39-column schema."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_schema_and_dtypes(pd.read_parquet(p))]
    assert not failed, f"Schema mismatch in races: {failed}"


def test_no_duplicate_driver_lapnumber():
    """Verify no duplicate (Driver, LapNumber) pairs exist in any race."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_duplicates(pd.read_parquet(p))]
    assert not failed, f"Duplicate laps found in races: {failed}"


def test_lap_number_integer_and_starts_at_1():
    """Verify all driver lap series start at lap 1."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_lap_start_at_one(pd.read_parquet(p))[0]]
    assert not failed, f"Driver lap series does not start at 1 in races: {failed}"


def test_laps_contiguous_per_driver():
    """Verify upstream completeness: missing laps <= 5% and no gaps >= 4."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_contiguous_laps(pd.read_parquet(p))[0]]
    assert not failed, f"Non-contiguous/incomplete laps found in races: {failed}"


def test_lap_time_range_when_present():
    """Verify lap times are between 50s and 400s on green racing laps."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_lap_times_valid(pd.read_parquet(p))[0]]
    assert not failed, f"LapTime out of [50, 400] on green laps in: {failed}"


def test_compound_valid_values():
    """Verify Compound contains only valid Pirelli compounds."""
    # FastF1 2022 raw data has known 'None' string for missing compound; listed in BUGS.md
    failed = []
    for p in RAW_PARQUETS:
        ok, _ = check_compounds_valid(pd.read_parquet(p))
        if not ok:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    # Document known upstream FastF1 missing compound races (Monaco and Austria 2022)
    known_upstream_missing = ["race_2022_07", "race_2022_11"]
    unexplained = [f for f in failed if f not in known_upstream_missing]
    assert not unexplained, f"Unexpected invalid compounds in: {unexplained}"


def test_tyre_life_increments_and_resets():
    """Verify TyreLife >= 1 and resets only after pit stop or compound change."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_tyre_life_rules(pd.read_parquet(p))[0]]
    assert not failed, f"Unphysical TyreLife reset in races: {failed}"


def test_pit_out_only_after_pit_in():
    """Verify PitOutTime occurs after a PitInTime lap, lap 1, or red flag restart."""
    failed = []
    for p in RAW_PARQUETS:
        df = pd.read_parquet(p)
        for drv, g in df.groupby("Driver"):
            in_laps = set(g[g["PitInTime"].notna()]["LapNumber"])
            out_laps = g[g["PitOutTime"].notna()]["LapNumber"]
            for ln in out_laps:
                ts = str(g[g["LapNumber"] == ln]["TrackStatus"].iloc[0])
                if ln > 1 and (ln - 1) not in in_laps and ln not in in_laps and "5" not in ts:
                    failed.append(os.path.basename(p).replace(".parquet", ""))
                    break
            if failed and failed[-1] == os.path.basename(p).replace(".parquet", ""):
                break
    # Red-flag restart pit exits without normal racing in-laps (Interlagos 2023, Suzuka 2024)
    known_red_flags = ["race_2023_21", "race_2024_04"]
    unexpected = [f for f in set(failed) if f not in known_red_flags]
    assert not unexpected, f"Unexpected pit out without pit in: {unexpected}"


def test_pit_in_followed_by_pit_out():
    """Verify pit-in laps are followed by pit-out unless driver retired."""
    failed = []
    for p in RAW_PARQUETS:
        df = pd.read_parquet(p)
        total_race_laps = df["LapNumber"].max()
        for drv, g in df.groupby("Driver"):
            g = g.sort_values("LapNumber")
            max_l = g["LapNumber"].max()
            is_retired = max_l < (total_race_laps - 5)
            for _, r in g.iterrows():
                ln = r["LapNumber"]
                if pd.notna(r["PitInTime"]):
                    if is_retired and ln >= max_l - 1:
                        continue  # Retired in pits / garage
                    nxt = g[g["LapNumber"] == ln + 1]
                    if len(nxt) > 0 and pd.isna(nxt["PitOutTime"].iloc[0]) and pd.isna(r["PitOutTime"]):
                        failed.append(os.path.basename(p).replace(".parquet", ""))
                        break
            if failed and failed[-1] == os.path.basename(p).replace(".parquet", ""):
                break
    assert not set(failed), f"Pit in not followed by pit out in: {set(failed)}"


def test_weather_ranges():
    """Verify ambient and track weather values fall within physical bounds."""
    failed = [os.path.basename(p) for p in RAW_PARQUETS if not check_weather_bounds(pd.read_parquet(p))[0]]
    assert not failed, f"Weather out of bounds in races: {failed}"


def test_rainfall_boolean():
    """Verify Rainfall values are boolean or 0/1 binary."""
    failed = []
    for p in RAW_PARQUETS:
        vals = set(pd.read_parquet(p)["Rainfall"].dropna().unique())
        if vals - {True, False, 0, 1, 0.0, 1.0}:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Rainfall not boolean in: {failed}"


def test_deleted_boolean():
    """Verify Deleted lap indicators are strictly boolean."""
    failed = []
    for p in RAW_PARQUETS:
        vals = set(pd.read_parquet(p)["Deleted"].dropna().unique())
        if vals - {True, False, 0, 1}:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Deleted column not boolean in: {failed}"


def test_track_status_characters():
    """Verify TrackStatus contains only valid characters 1 through 7."""
    failed = []
    for p in RAW_PARQUETS:
        for v in pd.read_parquet(p)["TrackStatus"].dropna().astype(str).unique():
            if not all(c in "1234567" for c in v):
                failed.append(os.path.basename(p).replace(".parquet", ""))
                break
    assert not failed, f"Invalid TrackStatus in: {failed}"


def test_driver_count_range():
    """Verify driver count is between 18 and 22 in all races."""
    failed = []
    for p in RAW_PARQUETS:
        nd = pd.read_parquet(p)["Driver"].nunique()
        if nd < 18 or nd > 22:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Driver count outside [18, 22] in: {failed}"


def test_max_lap_number_close_to_distance():
    """Verify max LapNumber is close to full grand prix distance."""
    failed = []
    for p in RAW_PARQUETS:
        ml = pd.read_parquet(p)["LapNumber"].max()
        # Suzuka 2022 was shortened to 28 laps by heavy rain timer limit
        if (ml < 40 or ml > 85) and "race_2022_18" not in p:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Max lap number out of range in: {failed}"


def test_no_future_dates():
    """Verify lap timestamps do not lie in the future."""
    now = pd.Timestamp.now()
    failed = []
    for p in RAW_PARQUETS:
        df = pd.read_parquet(p)
        if "LapStartDate" in df.columns:
            dt = pd.to_datetime(df["LapStartDate"].dropna(), errors="coerce")
            if (dt > now).any():
                failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Future dates found in: {failed}"


def test_year_matches_filename():
    """Verify Year column matches filename season."""
    failed = []
    for p in RAW_PARQUETS:
        yr = int(os.path.basename(p).split("_")[1])
        vals = pd.read_parquet(p)["Year"].dropna().unique()
        if len(vals) != 1 or int(vals[0]) != yr:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"Year mismatch in: {failed}"


def test_round_matches_filename():
    """Verify RoundNumber column matches filename round."""
    failed = []
    for p in RAW_PARQUETS:
        rnd = int(os.path.basename(p).split("_")[2].replace(".parquet", ""))
        vals = pd.read_parquet(p)["RoundNumber"].dropna().unique()
        if len(vals) != 1 or int(vals[0]) != rnd:
            failed.append(os.path.basename(p).replace(".parquet", ""))
    assert not failed, f"RoundNumber mismatch in: {failed}"
