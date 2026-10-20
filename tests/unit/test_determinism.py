import os
import hashlib
import pytest
import pandas as pd
from tests.fixtures.sample_race_builder import build_mock_openf1_data
from src.openf1_convert import convert_openf1_bundle


def compute_df_hash(df):
    """Compute sha256 of dataframe csv representation."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def test_converter_determinism_multiple_runs():
    """Re-converting 10 races twice produces identical data hashes."""
    for i in range(10):
        data = build_mock_openf1_data(season=2023, round_num=i+1, session_key=9100+i)
        df1 = convert_openf1_bundle(data, {"event_name": f"GP_{i}"}, i + 1, 2023)
        df2 = convert_openf1_bundle(data, {"event_name": f"GP_{i}"}, i + 1, 2023)
        h1 = compute_df_hash(df1)
        h2 = compute_df_hash(df2)
        assert h1 == h2, f"Non-deterministic conversion for race {i+1}"


def test_converter_row_order_invariance():
    """Input laps in reverse or shuffled order produce identical set of records."""
    data = build_mock_openf1_data(season=2023, round_num=1)
    df_orig = convert_openf1_bundle(data, {"event_name": "Sakhir"}, 1, 2023)
    shuffled_data = build_mock_openf1_data(season=2023, round_num=1)
    shuffled_data["laps"] = shuffled_data["laps"][::-1]
    df_shuffled = convert_openf1_bundle(shuffled_data, {"event_name": "Sakhir"}, 1, 2023)
    assert len(df_orig) == len(df_shuffled)
    s1 = df_orig.sort_values(["DriverNumber", "LapNumber"]).reset_index(drop=True)
    s2 = df_shuffled.sort_values(["DriverNumber", "LapNumber"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(s1, s2)


def test_cached_run_makes_zero_network_calls(monkeypatch, tmp_path):
    """Verify loading from cache performs zero network requests."""
    from src.openf1_source import fetch_openf1_endpoint, get_cache_path
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    endpoint = "sessions"
    query = "year=2023"
    cache_file = get_cache_path(str(cache_dir), endpoint, query)
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write('[{"session_key": 9999, "circuit_short_name": "Test"}]')
    def fail_request(*args, **kwargs):
        raise AssertionError("Network called when cache was available!")
    monkeypatch.setattr("requests.get", fail_request)
    data = fetch_openf1_endpoint(endpoint, query, str(cache_dir))
    assert len(data) == 1
    assert data[0]["session_key"] == 9999
