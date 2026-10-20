import os
import time
import json
import requests
import pandas as pd
from src.openf1_convert import convert_openf1_bundle
from src.validate_raw import validate_race_dataframe

BASE_URL = "https://api.openf1.org/v1"


def get_cache_path(cache_dir, endpoint, identifier):
    """Return disk cache file path for an endpoint and identifier."""
    safe_id = str(identifier).replace("=", "_").replace("&", "_").replace("?", "_")
    return os.path.join(cache_dir, f"{endpoint}_{safe_id}.json")


def read_cached_json(file_path):
    """Read and return JSON data from local cache if file exists."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def write_cached_json(file_path, data):
    """Save data to local disk cache as JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def request_with_retry(url, wait_fn=time.sleep):
    """Execute HTTP GET with 429 rate limit backoff and transient retries."""
    for attempt in range(1, 4):
        wait_fn(0.5)
        try:
            resp = requests.get(url, timeout=25)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return []
            if resp.status_code == 429:
                if attempt == 3:
                    raise RuntimeError("HTTP 429: Rate limit exceeded after 3 retries")
                wait_fn(30)
                continue
            if attempt == 3:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:100]}")
            wait_fn(2)
        except (requests.Timeout, requests.ConnectionError) as err:
            if attempt == 3:
                raise RuntimeError(f"Network error: {str(err)}")
            wait_fn(2)
    return None


def fetch_openf1_endpoint(endpoint, query_str, cache_dir, wait_fn=time.sleep):
    """Fetch OpenF1 endpoint using cache first and rate-limited HTTP fallback."""
    cache_path = get_cache_path(cache_dir, endpoint, query_str)
    cached = read_cached_json(cache_path)
    if cached is not None:
        return cached
    url = f"{BASE_URL}/{endpoint}?{query_str}"
    data = request_with_retry(url, wait_fn=wait_fn)
    if data is not None:
        write_cached_json(cache_path, data)
    return data


def get_session_date_start(session):
    """Extract start date string from session dictionary for sorting."""
    return str(session.get("date_start", ""))


def fetch_race_sessions(year, cache_dir="data/openf1_cache", wait_fn=time.sleep):
    """Fetch all race sessions for a given year sorted chronologically."""
    query = f"year={year}&session_name=Race"
    data = fetch_openf1_endpoint("sessions", query, cache_dir, wait_fn=wait_fn)
    if not isinstance(data, list):
        return []
    data.sort(key=get_session_date_start)
    return data


def fetch_race_bundle(session_key, cache_dir="data/openf1_cache", wait_fn=time.sleep):
    """Fetch laps, stints, weather, race control, and drivers for a session."""
    query = f"session_key={session_key}"
    endpoints = ["laps", "stints", "weather", "race_control", "drivers"]
    bundle = {}
    for ep in endpoints:
        data = fetch_openf1_endpoint(ep, query, cache_dir, wait_fn=wait_fn)
        bundle[ep] = data if isinstance(data, list) else []
    return bundle


def get_openf1_past_schedule(year, cache_dir="data/openf1_cache", wait_fn=time.sleep):
    """Fetch past race events for a season using OpenF1 sessions."""
    sessions = fetch_race_sessions(year, cache_dir=cache_dir, wait_fn=wait_fn)
    past = []
    now = pd.Timestamp.now().tz_localize(None)
    for idx, s in enumerate(sessions):
        d_val = s.get("date_start")
        if d_val:
            dt = pd.to_datetime(d_val, format="ISO8601").tz_localize(None)
            if dt <= now:
                event = dict(s)
                event["Year"] = int(year)
                event["RoundNumber"] = idx + 1
                event["EventName"] = s.get("circuit_short_name", "Grand Prix")
                event["EventDate"] = str(d_val)[:10]
                past.append(event)
    return past


def fetch_and_convert_openf1_race(event, round_num, year, cache_dir, wait_fn=time.sleep):
    """Fetch OpenF1 data bundle and convert to standard 39-column DataFrame."""
    sk = event.get("session_key")
    bundle = fetch_race_bundle(sk, cache_dir=cache_dir, wait_fn=wait_fn)
    rcm_count = len(bundle.get("race_control", []))
    df = convert_openf1_bundle(bundle, event, round_num, year)
    return df, rcm_count


def process_openf1_race(event, raw_dir, openf1_dir, wait_fn=time.sleep):
    """Fetch, convert, validate, and save a single OpenF1 race dataset."""
    year, r_num = int(event["Year"]), int(event["RoundNumber"])
    e_name = str(event.get("EventName", event.get("circuit_short_name", "Grand Prix")))
    e_date = str(event.get("EventDate", str(event.get("date_start", ""))[:10]))
    file_path = os.path.join(raw_dir, f"race_{year}_{r_num:02d}.parquet")
    try:
        df, rcm_cnt = fetch_and_convert_openf1_race(event, r_num, year, openf1_dir, wait_fn=wait_fn)
    except Exception as err:
        return {"season": year, "round": r_num, "event_name": e_name, "date": e_date,
                "status": f"failed: {str(err)}", "source": "openf1", "laps_rows": 0, "rcm_rows": 0,
                "deleted_known": False, "isaccurate_known": False, "rcm_note": "fetch failed", "error": str(err)}
    is_val, rsn = validate_race_dataframe(df, rcm_count=rcm_cnt, allow_missing_rcm=False)
    laps_cnt = len(df) if df is not None else 0
    if is_val and df is not None:
        os.makedirs(raw_dir, exist_ok=True)
        df.to_parquet(file_path, index=False)
        return {"season": year, "round": r_num, "event_name": e_name, "date": e_date,
                "status": "downloaded", "source": "openf1", "laps_rows": laps_cnt, "rcm_rows": rcm_cnt,
                "deleted_known": True, "isaccurate_known": False, "rcm_note": f"openf1 ({rcm_cnt} rcm rows)", "error": ""}
    return {"season": year, "round": r_num, "event_name": e_name, "date": e_date,
            "status": f"failed: {rsn}", "source": "openf1", "laps_rows": laps_cnt, "rcm_rows": rcm_cnt,
            "deleted_known": False, "isaccurate_known": False, "rcm_note": "openf1 validation failed", "error": rsn}
