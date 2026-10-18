import os
import time
import logging
import argparse
import yaml
import pandas as pd
import fastf1
from fastf1.exceptions import RateLimitExceededError
from src.validate_raw import (
    WarningHandler, validate_parquet_file, try_restore_from_old,
    get_past_race_schedule, fetch_and_validate
)
from src.openf1_source import get_openf1_past_schedule, process_openf1_race


def load_config(config_path="config.yaml"):
    """Read project configuration from yaml file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def enable_f1_cache(cache_dir="fastf1_cache"):
    """Enable FastF1 local disk cache to avoid redundant network calls."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def record_race_status(csv_path, record):
    """Update reports/races.csv, migrating old layouts and preserving rows."""
    cols = ["season", "round", "event_name", "date", "status", "source",
            "laps_rows", "rcm_rows", "deleted_known", "isaccurate_known", "rcm_note", "error"]
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df = pd.DataFrame(columns=cols)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not set(cols).issubset(set(df.columns)):
            backup = os.path.join(dir_name, "races_old_part1.csv") if dir_name else "races_old_part1.csv"
            if not os.path.exists(backup):
                os.replace(csv_path, backup)
            df = pd.DataFrame(columns=cols)
    mask = (df["season"] == record["season"]) & (df["round"] == record["round"])
    df = pd.concat([df[~mask], pd.DataFrame([record])], ignore_index=True)
    df.sort_values(["season", "round"]).to_csv(csv_path, index=False)


def fetch_race_with_retry(year, round_num, raw_dir, warn_handler, state, wait_fn=time.sleep):
    """Load race session with rate-limit retries and repeated error short-circuit."""
    rcm_fault = state.get("rcm_fault_count", 0) >= 3
    for attempt in range(1, 4):
        try:
            df, rcm_cnt, del_k, note, is_v, rsn, ep = fetch_and_validate(year, round_num, warn_handler, rcm_fault)
            if ep == "race_control_messages":
                state["rcm_fault_count"] = state.get("rcm_fault_count", 0) + 1
            if is_v:
                return df, rcm_cnt, del_k, note, "valid", ""
            if attempt == 3:
                restored = try_restore_from_old(year, round_num, raw_dir)
                if restored is not None:
                    return restored, 0, False, "restored from raw_old_no_messages", "valid", ""
                return None, rcm_cnt, del_k, note, f"failed: {ep or rsn}", ep or rsn
            wait_fn(30)
        except RateLimitExceededError as err:
            if attempt == 3:
                return None, 0, False, "rate limit exceeded", "failed", str(err)
            wait_fn(600)
        except Exception as err:
            if attempt == 3:
                return None, 0, False, "exception", "failed", str(err)
            wait_fn(30)


def process_cached_race(file_path, year, r_num, e_name, e_date, source, csv_path):
    """Handle status reporting for previously downloaded valid parquet file."""
    laps_df = pd.read_parquet(file_path)
    del_k = not laps_df["Deleted"].isna().all()
    acc_k = source == "fastf1" and not laps_df["IsAccurate"].isna().all()
    rec = {"season": year, "round": r_num, "event_name": e_name, "date": e_date,
           "status": "cached", "source": source, "laps_rows": len(laps_df), "rcm_rows": -1,
           "deleted_known": del_k, "isaccurate_known": acc_k, "rcm_note": "cached parquet", "error": ""}
    record_race_status(csv_path, rec)
    return rec


def process_single_race(event, raw_dir, csv_path, warn_handler, state, source="fastf1", openf1_dir="data/openf1_cache", wait_fn=time.sleep):
    """Handle validation, caching, downloading, and logging for one race round."""
    year = int(event["Year"]) if "Year" in event else int(str(event["EventDate"])[:4])
    r_num = int(event["RoundNumber"])
    e_name = str(event.get("EventName", event.get("circuit_short_name", "Grand Prix")))
    e_date = str(event.get("EventDate", str(event.get("date_start", ""))[:10]))
    file_path = os.path.join(raw_dir, f"race_{year}_{r_num:02d}.parquet")
    if os.path.exists(file_path) and validate_parquet_file(file_path, allow_missing_rcm=True)[0]:
        return process_cached_race(file_path, year, r_num, e_name, e_date, source, csv_path)
    if source == "openf1":
        rec = process_openf1_race(event, raw_dir, openf1_dir, wait_fn=wait_fn)
        record_race_status(csv_path, rec)
        return rec
    data, rcm_cnt, del_k, note, status, err = fetch_race_with_retry(year, r_num, raw_dir, warn_handler, state, wait_fn)
    laps = len(data) if data is not None else 0
    if status == "valid" and data is not None:
        os.makedirs(raw_dir, exist_ok=True)
        data.to_parquet(file_path, index=False)
        status = "downloaded"
    rec = {"season": year, "round": r_num, "event_name": e_name, "date": e_date,
           "status": status, "source": "fastf1", "laps_rows": laps, "rcm_rows": rcm_cnt,
           "deleted_known": del_k, "isaccurate_known": True, "rcm_note": note, "error": err}
    record_race_status(csv_path, rec)
    return rec


def check_circuit_breaker(rec, state, wait_fn=time.sleep):
    """Only count failures where laps themselves failed to load towards breaker."""
    if rec["status"].startswith("failed") and rec["laps_rows"] == 0:
        state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
        if state["consecutive_failures"] >= 3:
            state["breaker_waits"] = state.get("breaker_waits", 0) + 1
            state["consecutive_failures"] = 0
            if state["breaker_waits"] > 3:
                print("API UNAVAILABLE")
                return False
            wait_fn(1200)
    else:
        state["consecutive_failures"] = 0
    return True


def run_download_loop(years, raw_dir, csv_path, config, wait_fn=time.sleep):
    """Run sequential downloads across seasons with circuit breaker protection."""
    state = {"consecutive_failures": 0, "breaker_waits": 0, "rcm_fault_count": 0}
    warn_handler = WarningHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)
    of1_dir = config["paths"].get("openf1_cache_dir", "data/openf1_cache")
    sources = config.get("data_sources", {})
    for y in years:
        src = sources.get(y, sources.get(str(y), "fastf1"))
        schedule = get_openf1_past_schedule(y, of1_dir, wait_fn) if src == "openf1" else get_past_race_schedule(y)
        if isinstance(schedule, pd.DataFrame):
            events = [row.to_dict() for _, row in schedule.iterrows()]
        else:
            events = schedule
        for event in events:
            rec = process_single_race(event, raw_dir, csv_path, warn_handler, state, src, of1_dir, wait_fn)
            msg = f"[{rec['season']} R{rec['round']:02d} {rec['event_name']}] {rec['status']} ({rec['laps_rows']} laps, {rec['source']})"
            print(msg, flush=True)
            if not check_circuit_breaker(rec, state, wait_fn):
                return state
    return state


def main():
    """CLI entry point for downloading race datasets."""
    parser = argparse.ArgumentParser(description="Download F1 race data")
    parser.add_argument("--season", type=str, default="all", help="Season year (e.g. 2022) or 'all'")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    enable_f1_cache(config["paths"].get("cache_dir", "fastf1_cache"))
    raw_dir, rep_dir = config["paths"]["raw_dir"], config["paths"]["reports_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    csv_path = os.path.join(rep_dir, "races.csv")
    years = [2022, 2023, 2024, 2025] if args.season.lower() == "all" else [int(args.season)]
    run_download_loop(years, raw_dir, csv_path, config)


if __name__ == "__main__":
    main()
