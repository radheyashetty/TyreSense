import os
import pandas as pd
from src.validate_raw import get_part1_columns, check_race_completeness
from src.check_raw import inspect_raw_parquet, check_all_raw_parquets


def make_test_parquet(file_path, null_lt=False, null_deleted=False):
    """Build a synthetic 39-column parquet fixture for testing."""
    cols = get_part1_columns()
    data = {c: [None] * 10 for c in cols}
    data["Driver"] = ["VER"] * 10
    data["LapNumber"] = list(range(1, 11))
    data["LapTime"] = [None] * 5 + [pd.Timedelta(seconds=90.0)] * 5 if null_lt else [pd.Timedelta(seconds=90.0)] * 10
    data["Compound"] = ["HARD"] * 10
    data["TyreLife"] = list(range(1, 11))
    data["TrackStatus"] = ["1"] * 10
    data["Deleted"] = [None] * 10 if null_deleted else [False] * 8 + [True] * 2
    data["TrackTemp"] = [35.0] * 10
    data["AirTemp"] = [25.0] * 10
    data["Humidity"] = [40.0] * 10
    data["Rainfall"] = [False] * 10
    data["EventName"] = ["Test GP"] * 10
    data["Location"] = ["Test Circuit"] * 10
    data["RoundNumber"] = [1] * 10
    data["Year"] = [2023] * 10
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path, index=False)
    return file_path


def test_inspect_valid_parquet(tmp_path):
    """Verify that a valid parquet passes inspection and records Deleted counts."""
    parquet_path = os.path.join(str(tmp_path), "race_2023_01.parquet")
    make_test_parquet(parquet_path)
    res = inspect_raw_parquet(parquet_path)
    assert res["valid"] == "yes"
    assert res["schema_match"] == "yes"
    assert "True:2" in res["deleted_counts"]
    assert "False:8" in res["deleted_counts"]


def test_inspect_flags_missing_lap_times(tmp_path):
    """Verify that parquets with < 80% non-null lap times are flagged."""
    parquet_path = os.path.join(str(tmp_path), "race_2023_02.parquet")
    make_test_parquet(parquet_path, null_lt=True)
    res = inspect_raw_parquet(parquet_path)
    assert res["valid"] == "no"
    assert "LapTime non-null < 80%" in res["reason"]


def test_inspect_flags_deleted_entirely_null(tmp_path):
    """Verify that parquets with completely null Deleted column are flagged."""
    parquet_path = os.path.join(str(tmp_path), "race_2023_03.parquet")
    make_test_parquet(parquet_path, null_deleted=True)
    res = inspect_raw_parquet(parquet_path)
    assert res["valid"] == "no"
    assert "Deleted column entirely null" in res["reason"]


def test_check_all_raw_parquets_creates_report(tmp_path):
    """Verify that check_all_raw_parquets saves a complete raw_check.csv."""
    raw_dir = str(tmp_path / "raw")
    reports_dir = str(tmp_path / "reports")
    make_test_parquet(os.path.join(raw_dir, "race_2023_01.parquet"))
    make_test_parquet(os.path.join(raw_dir, "race_2023_02.parquet"), null_lt=True)

    df_out = check_all_raw_parquets(raw_dir=raw_dir, reports_dir=reports_dir)
    csv_path = os.path.join(reports_dir, "raw_check.csv")
    assert os.path.exists(csv_path)
    assert len(df_out) == 2
    assert "deleted_counts" in df_out.columns
    assert "valid" in df_out.columns


def test_completeness_flags_missing_laps():
    """Verify that a race missing laps 2-25 for all drivers is flagged."""
    drivers = ["VER", "HAM"]
    rows = []
    for d in drivers:
        for lap in [1] + list(range(26, 58)):
            rows.append({"Driver": d, "LapNumber": float(lap)})
    df = pd.DataFrame(rows)
    is_comp, miss_share, rsn = check_race_completeness(df)
    assert not is_comp
    assert miss_share > 0.05
    assert "incomplete upstream feed" in rsn


def test_completeness_passes_normal_race_with_retirement():
    """Verify that a race where a driver legitimately retires early is not flagged."""
    rows = []
    for lap in range(1, 51):
        rows.append({"Driver": "VER", "LapNumber": float(lap)})
    for lap in range(1, 16):
        rows.append({"Driver": "LEC", "LapNumber": float(lap)})
    df = pd.DataFrame(rows)
    is_comp, miss_share, rsn = check_race_completeness(df)
    assert is_comp
    assert miss_share == 0.0
    assert rsn == "complete"
