import os
import socket
import pandas as pd
import numpy as np
import src.openf1_source as src_f1
from src.openf1_source import fetch_openf1_endpoint, request_with_retry
from src.openf1_convert import (
    parse_deleted_laps, parse_sc_intervals, compute_lap_track_status,
    build_stint_lookup, merge_weather_data, convert_openf1_bundle
)
from src.validate_raw import get_part1_columns


def make_sample_bundle():
    """Build mock OpenF1 bundle dictionaries for unit testing."""
    laps = [
        {"driver_number": 1, "lap_number": 1, "lap_duration": 90.0, "is_pit_out_lap": False,
         "date_start": "2023-03-05T15:00:00Z", "duration_sector_1": 30.0, "duration_sector_2": 30.0,
         "duration_sector_3": 30.0, "i1_speed": 250.0, "i2_speed": 260.0, "st_speed": 300.0},
        {"driver_number": 1, "lap_number": 2, "lap_duration": 91.0, "is_pit_out_lap": False,
         "date_start": "2023-03-05T15:01:30Z", "duration_sector_1": 30.0, "duration_sector_2": 31.0,
         "duration_sector_3": 30.0, "i1_speed": 250.0, "i2_speed": 260.0, "st_speed": 300.0},
        {"driver_number": 1, "lap_number": 3, "lap_duration": 110.0, "is_pit_out_lap": True,
         "date_start": "2023-03-05T15:03:01Z", "duration_sector_1": 40.0, "duration_sector_2": 40.0,
         "duration_sector_3": 30.0, "i1_speed": 200.0, "i2_speed": 210.0, "st_speed": 250.0}
    ]
    stints = [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 2, "compound": "SOFT", "tyre_age_at_start": 3},
              {"driver_number": 1, "stint_number": 2, "lap_start": 3, "lap_end": 3, "compound": "HARD", "tyre_age_at_start": 0}]
    weather = [{"date": "2023-03-05T15:01:30Z", "track_temperature": 32.5, "air_temperature": 24.0, "humidity": 45.0, "rainfall": 0}]
    race_control = [
        {"message": "CAR 1 (VER) TIME 1:30.000 DELETED - TRACK LIMITS AT TURN 4 LAP 2 18:05:00"},
        {"date": "2023-03-05T15:01:00Z", "message": "SAFETY CAR DEPLOYED"},
        {"date": "2023-03-05T15:02:00Z", "message": "TRACK CLEAR"}
    ]
    drivers = [{"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"}]
    return {"laps": laps, "stints": stints, "weather": weather, "race_control": race_control, "drivers": drivers}


def test_tyrelife_formula():
    """Verify TyreLife = tyre_age_at_start + (lap_number - lap_start) + 1."""
    stints = [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 5, "compound": "SOFT", "tyre_age_at_start": 3}]
    lookup = build_stint_lookup(stints)
    assert lookup[(1, 1)][2] == 4.0
    assert lookup[(1, 2)][2] == 5.0
    assert lookup[(1, 5)][2] == 8.0


def test_deleted_lap_parsing():
    """Verify deleted lap parsing extracts car number, lap, and reason."""
    rc = [{"message": "CAR 4 (NOR) TIME 1:39.777 DELETED - TRACK LIMITS AT TURN 4 LAP 3 18:07:34"}]
    del_map = parse_deleted_laps(rc)
    assert (4, 3) in del_map
    assert "TRACK LIMITS" in del_map[(4, 3)]


def test_sc_track_status_intervals():
    """Verify SC intervals mark overlapping laps with 4 and clear with 1."""
    rc = [{"date": "2023-03-05T15:01:00Z", "message": "SAFETY CAR DEPLOYED"},
          {"date": "2023-03-05T15:02:00Z", "message": "TRACK CLEAR"}]
    intervals = parse_sc_intervals(rc)
    t1 = pd.to_datetime("2023-03-05T15:00:00Z")
    t2 = pd.to_datetime("2023-03-05T15:01:30Z")
    t3 = pd.to_datetime("2023-03-05T15:03:00Z")
    assert compute_lap_track_status(t1, t2, intervals) == "4"
    assert compute_lap_track_status(t2, t3, intervals) == "4"
    assert compute_lap_track_status(t3, t3 + pd.Timedelta(seconds=90), intervals) == "1"


def test_pit_flags_and_schema():
    """Verify in-lap, out-lap flags, and 39-column schema generation."""
    bundle = make_sample_bundle()
    event_info = {"circuit_short_name": "Sakhir", "location": "Bahrain"}
    df = convert_openf1_bundle(bundle, event_info, round_number=1, year=2023)
    assert list(df.columns) == get_part1_columns()
    assert len(df) == 3
    assert pd.isna(df.loc[0, "PitInTime"]) and pd.isna(df.loc[0, "PitOutTime"])
    assert pd.notna(df.loc[1, "PitInTime"]) and pd.isna(df.loc[1, "PitOutTime"])
    assert pd.isna(df.loc[2, "PitInTime"]) and pd.notna(df.loc[2, "PitOutTime"])
    assert bool(df.loc[1, "Deleted"]) is True
    assert df.loc[0, "TrackTemp"] == 32.5


def test_rate_limit_429_retry(monkeypatch):
    """Verify that HTTP 429 triggers a 30-second wait and retry."""
    waits, calls = [], [0]

    class MockResponse:
        def __init__(self, code):
            self.status_code = code
            self.text = "Too many requests"

        def json(self):
            return [{"status": "ok"}]

    def mock_get(url, timeout=25):
        calls[0] += 1
        if calls[0] == 1:
            return MockResponse(429)
        return MockResponse(200)

    def record_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(src_f1.requests, "get", mock_get)
    res = request_with_retry("https://api.openf1.org/v1/test", wait_fn=record_wait)
    assert res == [{"status": "ok"}]
    assert 30 in waits


def test_cached_responses_make_no_network_call(tmp_path, monkeypatch):
    """Verify cached JSON on disk skips network request entirely."""
    cache_dir = str(tmp_path / "openf1_cache")
    bundle = make_sample_bundle()
    src_f1.write_cached_json(os.path.join(cache_dir, "test_q1.json"), bundle["laps"])

    def block_network(*args, **kwargs):
        raise RuntimeError("Network should not be contacted when cached")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    data = fetch_openf1_endpoint("test", "q1", cache_dir)
    assert len(data) == 3
