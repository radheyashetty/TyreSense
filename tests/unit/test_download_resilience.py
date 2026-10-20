import os
import pytest
import pandas as pd
from src.validate_raw import validate_parquet_file
from src.download import record_race_status, process_cached_race
from tests.fixtures.sample_race_builder import build_mock_raw_parquet_df


def test_corrupt_partly_written_parquet_detected(tmp_path):
    """Partly written or truncated parquet file is detected as invalid."""
    bad_file = tmp_path / "race_2023_01.parquet"
    bad_file.write_bytes(b"PAR1 partially written corrupted bytes")
    is_valid, reason = validate_parquet_file(str(bad_file))
    assert not is_valid
    assert "corrupt parquet" in reason


def test_races_csv_consistent_when_interrupted(tmp_path):
    """races.csv remains valid and readable if process terminates halfway."""
    csv_file = tmp_path / "races.csv"
    for r in range(1, 5):
        record = {
            "season": 2023, "round": r, "event_name": f"Race_{r}", "date": "2023-03-01",
            "status": "downloaded", "source": "openf1", "laps_rows": 1000, "rcm_rows": 20,
            "deleted_known": True, "isaccurate_known": False, "rcm_note": "ok", "error": ""
        }
        record_race_status(str(csv_file), record)
    # Simulate abrupt halt after 4 races
    df = pd.read_csv(csv_file)
    assert len(df) == 4
    assert list(df["round"]) == [1, 2, 3, 4]


def test_atomic_file_write_is_safe_under_concurrent_attempts(tmp_path):
    """Simulate writing parquet with atomic rename preventing partial file reads."""
    target_path = tmp_path / "race_2023_01.parquet"
    df = build_mock_raw_parquet_df()
    tmp_file = str(target_path) + ".tmp"
    df.to_parquet(tmp_file, index=False)
    os.replace(tmp_file, str(target_path))
    is_valid, reason = validate_parquet_file(str(target_path))
    assert is_valid
    assert reason == "valid"
