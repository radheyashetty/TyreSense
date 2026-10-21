import os
import json
import pytest
import pandas as pd
import numpy as np
from tests.fixtures.sample_race_builder import build_mock_openf1_data
from src.openf1_convert import convert_openf1_bundle
from src.openf1_source import read_cached_json, process_openf1_race, request_with_retry


def test_converter_empty_bundle():
    """Empty response produces empty standard schema dataframe."""
    df = convert_openf1_bundle({}, {"event_name": "Test"}, 1, 2023)
    assert len(df) == 0
    assert len(df.columns) == 39


def test_converter_missing_keys():
    """Bundle missing optional keys converts without error."""
    bundle = {"laps": [{"driver_number": 1, "lap_number": 1, "lap_duration": 90.0, "date_start": "2023-03-05T15:00:00Z"}]}
    df = convert_openf1_bundle(bundle, {}, 1, 2023)
    assert len(df) == 1
    assert "LapTime" in df.columns


def test_converter_null_lap_duration():
    """Null lap_duration creates NaT LapTime instead of crashing."""
    data = build_mock_openf1_data()
    data["laps"][0]["lap_duration"] = None
    df = convert_openf1_bundle(data, {}, 1, 2023)
    assert pd.isna(df.loc[0, "LapTime"])


def test_converter_laps_out_of_order_and_duplicates():
    """Laps with duplicates are deduped safely without crashing."""
    data = build_mock_openf1_data()
    data["laps"] = data["laps"] + [data["laps"][0]]
    df = convert_openf1_bundle(data, {}, 1, 2023)
    d1 = df[df["Driver"] == "VER"]["LapNumber"].tolist()
    assert d1 == sorted(d1)
    assert len(d1) == len(set(d1))


def test_converter_stint_end_before_start_and_overlapping():
    """Stints with lap_end < lap_start or overlapping stints resolve safely."""
    data = build_mock_openf1_data()
    data["stints"][0]["lap_end"] = 0  # < lap_start 1
    df = convert_openf1_bundle(data, {}, 1, 2023)
    assert len(df) > 0
    assert (df["TyreLife"] >= 1.0).all()


def test_converter_driver_with_no_stints():
    """Driver with no stint records falls back to default values."""
    data = build_mock_openf1_data()
    data["stints"] = []
    df = convert_openf1_bundle(data, {}, 1, 2023)
    assert len(df) > 0
    assert (df["TyreLife"] >= 1.0).all()


def test_converter_weather_before_first_lap():
    """Weather records before start of session are handled by merge_asof."""
    data = build_mock_openf1_data()
    for w in data["weather"]:
        w["date"] = "2020-01-01T00:00:00Z"
    df = convert_openf1_bundle(data, {}, 1, 2023)
    assert len(df) > 0
    assert "TrackTemp" in df.columns


def test_converter_unexpected_race_control_and_unicode():
    """Unexpected text in race control and unicode names do not crash."""
    data = build_mock_openf1_data()
    data["race_control"] = [{"message": "UNKNOWN MESSAGE 123", "date": "2023-03-05T15:05:00Z"}]
    data["drivers"][0]["name_acronym"] = "VÉR"
    df = convert_openf1_bundle(data, {}, 1, 2023)
    assert len(df) > 0
    assert "VÉR" in df["Driver"].values


def test_cache_corrupted_json_returns_none(tmp_path):
    """Corrupted cache file returns None so it can be refetched."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ incomplete json ...")
    assert read_cached_json(str(corrupt_file)) is None


def noop_wait(s):
    """No-op wait function for testing without sleep delays."""
    return None


def test_process_openf1_race_handles_errors_gracefully(tmp_path):
    """Pipeline fails race with clear reason rather than saving bad parquet."""
    event = {"Year": 2023, "RoundNumber": 1, "session_key": "invalid_key"}
    rec = process_openf1_race(event, str(tmp_path / "raw"), str(tmp_path / "cache"), wait_fn=noop_wait)
    assert rec["status"].startswith("failed")
    assert not os.path.exists(tmp_path / "raw" / "race_2023_01.parquet")
