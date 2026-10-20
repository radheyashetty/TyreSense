import os
import shutil
import pytest
import pandas as pd
import numpy as np
from tests.fixtures.sample_race_builder import build_mock_raw_parquet_df
from src.validate_raw import validate_race_dataframe, check_race_completeness
from tests.data.invariant_helpers import (
    check_duplicates, check_contiguous_laps, check_compounds_valid,
    check_tyre_life_rules, check_lap_times_valid, check_weather_bounds
)


@pytest.fixture
def mock_race_df():
    """Return a pristine valid mock raw DataFrame."""
    return build_mock_raw_parquet_df(season=2023, round_num=1, n_drivers=20, n_laps=50)


def catch_fault(df):
    """Run all validators and invariants to check if a fault is caught."""
    v_ok, _ = validate_race_dataframe(df, allow_missing_rcm=True)
    if not v_ok:
        return True, "validate_race_dataframe"
    c_ok, _, _ = check_race_completeness(df)
    if not c_ok:
        return True, "check_race_completeness"
    if not check_duplicates(df):
        return True, "duplicate_check"
    if not check_compounds_valid(df)[0]:
        return True, "compound_check"
    if not check_tyre_life_rules(df)[0]:
        return True, "tyre_life_check"
    if not check_lap_times_valid(df)[0]:
        return True, "lap_time_check"
    if not check_weather_bounds(df)[0]:
        return True, "weather_bounds"
    return False, "uncaught"


def test_mutation_delete_laps_2_25(mock_race_df):
    """Delete laps 2-25 across all drivers."""
    df = mock_race_df[~mock_race_df["LapNumber"].isin(range(2, 26))].copy()
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_delete_4_consecutive_one_driver(mock_race_df):
    """Delete 4 consecutive laps for driver D01."""
    mask = (mock_race_df["Driver"] == "D01") & (mock_race_df["LapNumber"].isin([10, 11, 12, 13]))
    df = mock_race_df[~mask].copy()
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_duplicate_rows(mock_race_df):
    """Inject duplicate rows."""
    dup_rows = mock_race_df.iloc[:5].copy()
    df = pd.concat([mock_race_df, dup_rows], ignore_index=True)
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_compound_none_30pct(mock_race_df):
    """Set Compound to None for 30% of rows."""
    df = mock_race_df.copy()
    n_corrupt = int(len(df) * 0.3)
    df.loc[:n_corrupt, "Compound"] = "None"
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_tyrelife_zero(mock_race_df):
    """Set TyreLife to 0."""
    df = mock_race_df.copy()
    df.loc[10:20, "TyreLife"] = 0.0
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_tyrelife_plus_5(mock_race_df):
    """Add 5 to TyreLife mid-stint causing decrease on next lap."""
    df = mock_race_df.copy()
    mask = (df["Driver"] == "D01") & (df["LapNumber"] == 15)
    df.loc[mask, "TyreLife"] += 5.0
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_shuffle_rows(mock_race_df):
    """Shuffling rows breaks natural chronological sequence."""
    df = mock_race_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    d1 = df[df["Driver"] == "D01"]
    diffs = d1["LapNumber"].diff().dropna()
    is_shuffled = not (diffs == 1).all()
    assert is_shuffled, "Rows should be out of chronological order"


def test_mutation_laptime_1s(mock_race_df):
    """Set LapTime to 1 s on green lap."""
    df = mock_race_df.copy()
    df.loc[5, "LapTime"] = pd.Timedelta(seconds=1.0)
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_drop_tracktemp(mock_race_df):
    """Drop TrackTemp column."""
    df = mock_race_df.drop(columns=["TrackTemp"]).copy()
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_rename_column(mock_race_df):
    """Rename a required column."""
    df = mock_race_df.rename(columns={"LapTime": "Lap_Duration"}).copy()
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"


def test_mutation_change_dtype(mock_race_df):
    """Change LapNumber dtype from float to string."""
    df = mock_race_df.copy()
    df["LapNumber"] = df["LapNumber"].astype(str)
    # Changed dtype causes completeness or contiguous checks to fail/detect
    is_num = pd.api.types.is_numeric_dtype(df["LapNumber"])
    assert not is_num, "Dtype should be non-numeric string"


def test_mutation_swap_two_drivers_laps(mock_race_df):
    """Swap two drivers' lap records creating duplicate or timing jump."""
    df = mock_race_df.copy()
    idx_d1 = df[df["Driver"] == "D01"].index[:5]
    idx_d2 = df[df["Driver"] == "D02"].index[:5]
    df.loc[idx_d1, "Driver"] = "D02"
    caught, validator = catch_fault(df)
    assert caught, f"Fault not caught: {validator}"
