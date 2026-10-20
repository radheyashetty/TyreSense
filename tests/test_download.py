import os
import socket
import logging
import pandas as pd
import fastf1
import src.download as dl
from src.download import record_race_status, process_single_race
from src.validate_raw import get_part1_columns


def make_weather_df():
    """Return mock weather dataframe for test session."""
    return pd.DataFrame({
        "TrackTemp": [35.0] * 5, "AirTemp": [25.0] * 5,
        "Humidity": [40.0] * 5, "Rainfall": [False] * 5
    })


def noop_wait(seconds):
    """Mock wait function doing nothing."""
    pass


def mock_probe_401(path):
    """Mock HTTP probe returning 401 status and 0 bytes."""
    return 401, 0


def mock_network_blocked(*args, **kwargs):
    """Mock socket connect to assert network is not contacted."""
    raise RuntimeError("Network called on cached race")


class DummySession:
    """Mock FastF1 session for unit testing."""
    def __init__(self, empty_laps=False, rcm_fails=False):
        self.empty_laps = empty_laps
        self.rcm_fails = rcm_fails
        self.load_messages_args = []
        self.api_path = "2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race/"
        self.event = {"EventName": "Test GP", "Location": "Track", "RoundNumber": 1, "year": 2099}
        self.laps = pd.DataFrame()
        self.race_control_messages = None

    def load(self, telemetry=False, weather=True, messages=True):
        self.load_messages_args.append(messages)
        if self.rcm_fails and messages:
            logger = logging.getLogger("fastf1")
            logger.warning("Failed to load race control messages!")
            self.race_control_messages = None
        else:
            self.race_control_messages = pd.DataFrame({"Message": ["Clear"] * 5})
        if self.empty_laps:
            self.laps = pd.DataFrame()
            return
        cols = get_part1_columns()
        data = {c: [None] * 5 for c in cols}
        data["Driver"], data["LapNumber"] = ["VER"] * 5, [1, 2, 3, 4, 5]
        data["LapTime"] = [pd.Timedelta(seconds=90.0)] * 5
        data["Compound"], data["TrackStatus"] = ["HARD"] * 5, ["1"] * 5
        data["TrackTemp"] = [35.0] * 5
        data["Deleted"] = [None] * 5 if self.rcm_fails else [False] * 5
        self.laps = pd.DataFrame(data)
        self.laps.get_weather_data = make_weather_df


def test_old_layout_races_csv_migration(tmp_path):
    """Verify that old-layout races.csv is safely migrated with all columns."""
    csv_path = os.path.join(str(tmp_path), "races.csv")
    old_df = pd.DataFrame({"season": [2022], "races_found": [22], "races_downloaded": [0], "races_skipped": [0]})
    old_df.to_csv(csv_path, index=False)
    rec = {"season": 2099, "round": 1, "event_name": "Bahrain GP", "date": "2099-03-05",
           "status": "downloaded", "laps_rows": 1056, "rcm_rows": 12, "deleted_known": True,
           "rcm_note": "rcm loaded", "error": ""}
    record_race_status(csv_path, rec)
    assert os.path.exists(os.path.join(str(tmp_path), "races_old_part1.csv"))
    new_df = pd.read_csv(csv_path)
    assert "deleted_known" in new_df.columns
    assert "rcm_note" in new_df.columns


def test_class_a_rcm_unavailable_success(tmp_path, monkeypatch):
    """Verify Class A race (RCM 401) saves parquet with deleted_known=False and breaker=0."""
    raw_dir, csv_path = str(tmp_path / "raw"), str(tmp_path / "races.csv")
    dummy = DummySession(rcm_fails=True)

    def mock_get_sess(y, r, t):
        return dummy

    monkeypatch.setattr(fastf1, "get_session", mock_get_sess)
    monkeypatch.setattr("src.validate_raw.probe_rcm_http", mock_probe_401)
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}
    event = {"Year": 2099, "RoundNumber": 10, "EventName": "Bahrain Grand Prix", "EventDate": "2099-03-05"}
    warn_handler = dl.WarningHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)

    rec = process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=noop_wait)
    dl.check_circuit_breaker(rec, state, wait_fn=noop_wait)
    assert rec["status"] == "downloaded"
    assert rec["deleted_known"] is False
    assert "HTTP 401" in rec["rcm_note"]
    assert state["consecutive_failures"] == 0
    assert os.path.exists(os.path.join(raw_dir, "race_2099_10.parquet"))


def test_laps_fail_completely_breaker_incremented(tmp_path, monkeypatch):
    """Verify empty laps fails validation and increments circuit breaker."""
    raw_dir, csv_path = str(tmp_path / "raw"), str(tmp_path / "races.csv")
    dummy = DummySession(empty_laps=True)

    def mock_get_sess(y, r, t):
        return dummy

    monkeypatch.setattr(fastf1, "get_session", mock_get_sess)
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}
    event = {"Year": 2099, "RoundNumber": 11, "EventName": "Bahrain Grand Prix", "EventDate": "2099-03-05"}
    warn_handler = dl.WarningHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)

    rec = process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=noop_wait)
    dl.check_circuit_breaker(rec, state, wait_fn=noop_wait)
    assert rec["status"].startswith("failed")
    assert state["consecutive_failures"] == 1
    assert not os.path.exists(os.path.join(raw_dir, "race_2099_11.parquet"))


def test_repeated_rcm_error_skips_messages_request(tmp_path, monkeypatch):
    """Verify 3 consecutive RCM faults causes 4th race to skip requesting messages."""
    raw_dir, csv_path = str(tmp_path / "raw"), str(tmp_path / "races.csv")
    sessions = {}

    def mock_get_sess(y, r, t):
        if r not in sessions:
            sessions[r] = DummySession(rcm_fails=True)
        return sessions[r]

    waits = []

    def record_wait(s):
        waits.append(s)

    monkeypatch.setattr(fastf1, "get_session", mock_get_sess)
    monkeypatch.setattr("src.validate_raw.probe_rcm_http", mock_probe_401)
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}
    warn_handler = dl.WarningHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)

    for i in range(1, 5):
        event = {"Year": 2099, "RoundNumber": 20 + i, "EventName": "Bahrain Grand Prix", "EventDate": "2099-03-05"}
        process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=record_wait)
    assert state["rcm_fault_count"] >= 3
    assert sessions[24].load_messages_args[0] is False
    assert 30 not in waits


def test_endpoint_name_recorded_in_error_field(tmp_path, monkeypatch):
    """Verify failing endpoint is recorded in error column when race fails."""
    raw_dir, csv_path = str(tmp_path / "raw"), str(tmp_path / "races.csv")
    dummy = DummySession(empty_laps=True, rcm_fails=True)

    def mock_get_sess(y, r, t):
        return dummy

    monkeypatch.setattr(fastf1, "get_session", mock_get_sess)
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}
    event = {"Year": 2099, "RoundNumber": 12, "EventName": "Bahrain Grand Prix", "EventDate": "2099-03-05"}
    warn_handler = dl.WarningHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)

    rec = process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=noop_wait)
    assert rec["error"] == "race_control_messages"


def test_cached_race_skips_network(tmp_path, monkeypatch):
    """Verify existing valid parquet skips network calls."""
    raw_dir, csv_path = str(tmp_path / "raw"), str(tmp_path / "races.csv")
    dummy = DummySession()

    def mock_get_sess(y, r, t):
        return dummy

    monkeypatch.setattr(fastf1, "get_session", mock_get_sess)
    event = {"Year": 2099, "RoundNumber": 13, "EventName": "Bahrain Grand Prix", "EventDate": "2099-03-05"}
    warn_handler = dl.WarningHandler()
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}

    rec1 = process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=noop_wait)
    assert rec1["status"] == "downloaded"
    monkeypatch.setattr(socket.socket, "connect", mock_network_blocked)
    rec2 = process_single_race(event, raw_dir, csv_path, warn_handler, state, wait_fn=noop_wait)
    assert rec2["status"] == "cached"
