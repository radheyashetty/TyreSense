import os
import logging
import requests
import pandas as pd
import fastf1
from fastf1.exceptions import DataNotLoadedError


class WarningHandler(logging.Handler):
    """Logging handler to capture warning messages from FastF1."""
    def __init__(self):
        super().__init__()
        self.warnings = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.warnings.append(record.getMessage())


def get_part1_columns():
    """Return the 39 expected column names from the canonical raw schema."""
    return ["Time", "Driver", "DriverNumber", "LapTime", "LapNumber", "Stint", "PitOutTime", "PitInTime",
            "Sector1Time", "Sector2Time", "Sector3Time", "Sector1SessionTime", "Sector2SessionTime",
            "Sector3SessionTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "Compound",
            "TyreLife", "FreshTyre", "Team", "LapStartTime", "LapStartDate", "TrackStatus", "Position",
            "Deleted", "DeletedReason", "FastF1Generated", "IsAccurate", "TrackTemp", "AirTemp", "Humidity",
            "Rainfall", "EventName", "Location", "RoundNumber", "Year"]


def probe_rcm_http(api_path):
    """Directly probe F1 live timing endpoint for race control messages."""
    path = api_path if not api_path.startswith("/static/") else api_path[len("/static/"):]
    url = f"https://livetiming.formula1.com/static/{path}RaceControlMessages.jsonStream"
    try:
        resp = requests.get(url, timeout=10)
        return resp.status_code, len(resp.content)
    except Exception:
        return -1, 0


def find_failing_endpoint(warnings):
    """Extract the first failing endpoint name from FastF1 warnings."""
    for w in warnings:
        if "Failed to load " in w:
            ep = w.split("Failed to load ")[1].replace("!", "").strip()
            return ep.replace(" ", "_").lower()
    return ""


def extract_race_data(session):
    """Extract laps, weather, and event metadata matching canonical telemetry schema."""
    laps = session.laps
    if laps is None or len(laps) == 0:
        return None
    weather = laps.get_weather_data()
    df = laps.copy()
    for col in ["TrackTemp", "AirTemp", "Humidity", "Rainfall"]:
        df[col] = weather[col].values
    ev = session.event
    df["EventName"], df["Location"], df["RoundNumber"] = ev["EventName"], ev["Location"], ev["RoundNumber"]
    df["Year"] = getattr(ev, "year", None) or ev.get("year", ev.get("Year", 2023))
    return df


def check_column_threshold(df, column_name, threshold):
    """Check if a column exists and its non-null share meets the threshold."""
    if column_name not in df.columns or len(df) == 0:
        return False
    return float(df[column_name].notna().mean()) >= threshold


def check_consecutive_missing(missing_laps, limit=4):
    """Check if any run of consecutive missing laps reaches or exceeds limit."""
    if len(missing_laps) < limit:
        return False
    sorted_l = sorted(missing_laps)
    streak = 1
    for i in range(1, len(sorted_l)):
        streak = (streak + 1) if sorted_l[i] == sorted_l[i - 1] + 1 else 1
        if streak >= limit:
            return True
    return False


def check_race_completeness(df):
    """Check if race has >5% missing laps or >=4 consecutive missing laps."""
    if df is None or len(df) == 0:
        return False, 1.0, "empty dataframe"
    tot_exp, tot_miss, has_consec = 0, 0, False
    for drv, g in df.groupby("Driver"):
        present = set(g["LapNumber"].dropna().astype(int))
        if not present:
            continue
        last_lap = max(present)
        expected = set(range(1, last_lap + 1))
        missing = expected - present
        tot_exp += len(expected)
        tot_miss += len(missing)
        if check_consecutive_missing(missing, 4):
            has_consec = True
    miss_share = (tot_miss / tot_exp) if tot_exp > 0 else 0.0
    if miss_share > 0.05 or has_consec:
        rsn = f"incomplete upstream feed ({miss_share * 100:.1f}% laps missing)"
        return False, miss_share, rsn
    return True, miss_share, "complete"


def validate_race_dataframe(df, rcm_count=None, allow_missing_rcm=False):
    """Validate extracted race dataframe against canonical raw data quality criteria."""
    if df is None or len(df) == 0:
        return False, "empty laps"
    if not set(get_part1_columns()).issubset(set(df.columns)):
        return False, "schema mismatch"
    for col, th in [("LapTime", 0.80), ("Compound", 0.90), ("TrackTemp", 0.90)]:
        if not check_column_threshold(df, col, th):
            return False, f"{col} non-null < {int(th*100)}%"
    if not allow_missing_rcm and rcm_count is not None and rcm_count < 1:
        return False, "race control messages < 1"
    is_comp, _, comp_rsn = check_race_completeness(df)
    if not is_comp:
        return False, comp_rsn
    return True, "valid"


def validate_parquet_file(file_path, allow_missing_rcm=True):
    """Validate a saved parquet file on disk against quality criteria."""
    if not os.path.exists(file_path):
        return False, "file not found"
    try:
        df = pd.read_parquet(file_path)
    except Exception as error:
        return False, f"corrupt parquet: {error}"
    is_valid, reason = validate_race_dataframe(df, rcm_count=None, allow_missing_rcm=allow_missing_rcm)
    if not is_valid:
        return False, reason
    return True, "valid"


def try_restore_from_old(year, r_num, raw_dir):
    """Restore cached parquet from data/raw_old_no_messages/ if valid."""
    old_file = os.path.join("data/raw_old_no_messages", f"race_{year}_{r_num:02d}.parquet")
    dest_file = os.path.join(raw_dir, f"race_{year}_{r_num:02d}.parquet")
    if os.path.exists(old_file) and validate_parquet_file(old_file, allow_missing_rcm=True)[0]:
        df = pd.read_parquet(old_file)
        os.makedirs(raw_dir, exist_ok=True)
        df.to_parquet(dest_file, index=False)
        return df
    return None


def is_past_race(event):
    """Return True if the race event date has already passed."""
    d = event.get("EventDate")
    return pd.to_datetime(d).tz_localize(None) <= pd.Timestamp.now().tz_localize(None) if pd.notna(d) else False


def get_past_race_schedule(year):
    """Fetch schedule for a season and filter to past non-testing races."""
    schedule = fastf1.get_event_schedule(year)
    valid_events = [ev for _, ev in schedule.iterrows() if ev.get("EventFormat") != "testing" and int(ev.get("RoundNumber", 0)) > 0 and is_past_race(ev)]
    return pd.DataFrame(valid_events) if valid_events else pd.DataFrame()


def load_session_with_handler(year, round_num, with_messages, warn_handler):
    """Load session and return session instance and captured warnings."""
    warn_handler.warnings.clear()
    session = fastf1.get_session(year, round_num, "R")
    try:
        session.load(telemetry=False, weather=True, messages=with_messages)
    except Exception as err:
        warn_handler.warnings.append(str(err))
    return session, list(warn_handler.warnings)


def fetch_and_validate(year, round_number, warn_handler, rcm_fault_fixed=False):
    """Load session, applying Class A HTTP probe fallback if RCM is unavailable."""
    with_messages = not rcm_fault_fixed
    session, warns = load_session_with_handler(year, round_number, with_messages, warn_handler)
    rcm_count = -1
    if with_messages:
        try:
            rcm_count = len(session.race_control_messages) if session.race_control_messages is not None else 0
        except (DataNotLoadedError, Exception):
            rcm_count = 0
    failing_ep = find_failing_endpoint(warns)
    if rcm_count > 0:
        df = extract_race_data(session)
        is_valid, reason = validate_race_dataframe(df, rcm_count=rcm_count, allow_missing_rcm=False)
        return df, rcm_count, True, f"rcm loaded ({rcm_count} rows)", is_valid, reason, failing_ep
    status_code, byte_len = probe_rcm_http(session.api_path)
    if status_code != 200 or byte_len == 0 or rcm_fault_fixed:
        if with_messages:
            session, warns = load_session_with_handler(year, round_number, False, warn_handler)
        df = extract_race_data(session)
        is_valid, reason = validate_race_dataframe(df, rcm_count=0, allow_missing_rcm=True)
        note = f"rcm unavailable at source (HTTP {status_code}, {byte_len} bytes)"
        return df, 0, False, note, is_valid, reason, failing_ep
    return None, 0, False, "rcm missing but probe 200", False, "rcm missing", failing_ep
