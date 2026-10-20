import os
import time
import json
import requests
import pandas as pd
from src.tracks import canonical_track_id

CACHE_DIR = "data/jolpica_cache"
BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "TyreSense/1.0 (research-audit)"}


def fetch_jolpica_json(endpoint, params=None):
    """Fetch JSON from Jolpica API with caching, rate limits, and 429 backoff."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    slug = endpoint.strip("/").replace("/", "_")
    p_str = "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items())) if params else ""
    cache_path = os.path.join(CACHE_DIR, f"{slug}{p_str}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(5):
        time.sleep(1.5)
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            if r.status_code == 429:
                time.sleep(60); continue
            if r.status_code == 404:
                return None
        except Exception as e:
            print(f"Request error for {url}: {e}, retry in 5s")
            time.sleep(5)
    return None


def fetch_all_race_laps(season, round_num):
    """Fetch all lap timings for a race across paginated responses."""
    offset, limit, all_laps = 0, 100, []
    while True:
        data = fetch_jolpica_json(f"{season}/{round_num}/laps.json", {"limit": limit, "offset": offset})
        if not data:
            break
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races or not races[0].get("Laps", []):
            break
        all_laps.extend(races[0].get("Laps", []))
        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break
    return all_laps


def fetch_race_pitstops(season, round_num):
    """Fetch all pitstops for a race."""
    data = fetch_jolpica_json(f"{season}/{round_num}/pitstops.json", {"limit": 100})
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("PitStops", []) if races else []


def fetch_race_results(season, round_num):
    """Fetch final race classification for driver completed lap counts."""
    data = fetch_jolpica_json(f"{season}/{round_num}/results.json", {"limit": 100})
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("Results", []) if races else []


def parse_time_seconds(time_str):
    """Convert mm:ss.fff or ss.fff string to float seconds."""
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        if ":" in time_str:
            p = time_str.split(":")
            return float(p[0]) * 60.0 + float(p[1])
        return float(time_str)
    except ValueError:
        return None


def get_jolpica_schedule(season):
    """Fetch Jolpica schedule and map canonical track_id to Ergast round number."""
    data = fetch_jolpica_json(f"{season}.json", {"limit": 100})
    if not data:
        return {}
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    mapping = {}
    for r in races:
        try:
            mapping[canonical_track_id(r.get("raceName", ""))] = int(r.get("round", 0))
        except ValueError:
            pass
    return mapping


def compare_race(race_id, season, round_num):
    """Compare one race against Jolpica reference data."""
    raw_path = f"data/raw/{race_id}.parquet"
    if not os.path.exists(raw_path):
        return {"race_id": race_id, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "file not in data/raw"}
    results = fetch_race_results(season, round_num)
    if not results:
        return {"race_id": race_id, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "no reference"}
    driver_map = {r.get("Driver", {}).get("driverId"): r.get("Driver", {}).get("code") for r in results if r.get("Driver", {}).get("code") and r.get("Driver", {}).get("driverId")}
    expected_laps = {r.get("Driver", {}).get("code"): int(r.get("laps", 0)) for r in results if r.get("Driver", {}).get("code")}
    raw_df = pd.read_parquet(raw_path)
    return evaluate_metrics(race_id, season, round_num, raw_df, driver_map, expected_laps)


def evaluate_metrics(race_id, season, round_num, raw_df, driver_map, expected_laps):
    """Evaluate lap times, pit stops, and completed laps against Jolpica."""
    all_laps = fetch_all_race_laps(season, round_num)
    if not all_laps:
        return {"race_id": race_id, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "no reference"}
    ref_laps = {}
    for lap in all_laps:
        lnum = int(lap.get("number", 0))
        for timing in lap.get("Timings", []):
            code = driver_map.get(timing.get("driverId"))
            sec = parse_time_seconds(timing.get("time"))
            if code and sec is not None:
                ref_laps[(code, lnum)] = sec
    total_cmp, match_lt = 0, 0
    flying_raw = raw_df[raw_df["LapNumber"] > 1]
    for _, row in flying_raw.iterrows():
        key = (row["Driver"], int(row["LapNumber"]))
        if key in ref_laps and pd.notna(row["LapTime"]):
            total_cmp += 1
            raw_sec = row["LapTime"].total_seconds()
            if abs(raw_sec - ref_laps[key]) <= 0.0051:
                match_lt += 1
    lt_share = (match_lt / total_cmp) if total_cmp > 0 else 0.0
    pit_share = evaluate_pitstops(season, round_num, raw_df, driver_map)
    drv_share = evaluate_driver_laps(raw_df, expected_laps)
    return {"race_id": race_id, "laps_compared": total_cmp, "laptime_match_share": round(lt_share, 4), "pit_lap_match_share": round(pit_share, 4), "driver_laps_match": round(drv_share, 4), "note": "ok"}


def evaluate_pitstops(season, round_num, raw_df, driver_map):
    """Compute pit stop match share between raw data and Jolpica."""
    pitstops = fetch_race_pitstops(season, round_num)
    ref_pits = {(driver_map.get(ps.get("driverId")), int(ps.get("lap", 0))) for ps in pitstops if driver_map.get(ps.get("driverId"))}
    raw_pits = {(r["Driver"], int(r["LapNumber"])) for _, r in raw_df[raw_df["PitInTime"].notna()].iterrows()}
    if not ref_pits and not raw_pits:
        return 1.0
    common = len(ref_pits.intersection(raw_pits))
    total = max(len(ref_pits), len(raw_pits))
    return common / total if total > 0 else 1.0


def evaluate_driver_laps(raw_df, expected_laps):
    """Check driver completed lap count against official classification."""
    matches, checked = 0, 0
    for drv, g in raw_df.groupby("Driver"):
        if drv in expected_laps:
            checked += 1
            if int(g["LapNumber"].max()) == expected_laps[drv]:
                matches += 1
    return matches / checked if checked > 0 else 1.0


def run_all_third_source_checks():
    """Run third source check for all races and output reports/testing/third_source.csv."""
    os.makedirs("reports/testing", exist_ok=True)
    df_races = pd.read_csv("reports/races.csv")
    valid_races = df_races[df_races["status"].isin(["downloaded", "cached"])]
    schedules = {yr: get_jolpica_schedule(yr) for yr in [2022, 2023, 2024, 2025]}
    records = []
    print(f"Starting Jolpica check on {len(valid_races)} valid races...", flush=True)
    for idx, (_, row) in enumerate(valid_races.iterrows()):
        yr, rnd = int(row["season"]), int(row["round"])
        rid = f"race_{yr}_{rnd:02d}"
        tid = row.get("track_id") if pd.notna(row.get("track_id")) else canonical_track_id(row["event_name"])
        j_rnd = schedules.get(yr, {}).get(tid)
        res = compare_race(rid, yr, j_rnd) if j_rnd else {"race_id": rid, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "not in reference schedule"}
        records.append(res)
        print(f"[{idx+1}/{len(valid_races)}] {rid}: cmp={res['laps_compared']}, lt={res['laptime_match_share']}, pit={res['pit_lap_match_share']}, note={res['note']}", flush=True)
        pd.DataFrame(records).to_csv("reports/testing/third_source.csv", index=False)
    print("Third-source verification complete. Saved to reports/testing/third_source.csv", flush=True)


if __name__ == "__main__":
    run_all_third_source_checks()
