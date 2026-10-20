import os
import time
import json
import requests
import pandas as pd

CACHE_DIR = "data/jolpica_cache"
BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "TyreSense/1.0 (research-audit)"}


def fetch_jolpica_json(endpoint, params=None):
    """Fetch JSON from Jolpica API with caching, rate limits, and 429 backoff."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    slug = endpoint.strip("/").replace("/", "_")
    p_str = ""
    if params:
        p_str = "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
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
                print(f"HTTP 429 rate limit on {url}, sleeping 60s (attempt {attempt + 1}/5)")
                time.sleep(60)
                continue
            if r.status_code == 404:
                return None
        except Exception as e:
            print(f"Request error for {url}: {e}, retry in 5s")
            time.sleep(5)
    return None


def fetch_all_race_laps(season, round_num):
    """Fetch all lap timings for a race across paginated responses."""
    offset = 0
    limit = 100
    all_laps = []
    while True:
        data = fetch_jolpica_json(f"{season}/{round_num}/laps.json", {"limit": limit, "offset": offset})
        if not data:
            break
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            break
        laps_page = races[0].get("Laps", [])
        if not laps_page:
            break
        all_laps.extend(laps_page)
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
    if not races:
        return []
    return races[0].get("PitStops", [])


def fetch_race_results(season, round_num):
    """Fetch final race classification for driver completed lap counts."""
    data = fetch_jolpica_json(f"{season}/{round_num}/results.json", {"limit": 100})
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    return races[0].get("Results", [])


def parse_time_seconds(time_str):
    """Convert mm:ss.fff or ss.fff string to float seconds."""
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(time_str)
    except ValueError:
        return None


def compare_race(race_id, season, round_num):
    """Compare one race against Jolpica reference data."""
    raw_path = f"data/raw/{race_id}.parquet"
    if not os.path.exists(raw_path):
        return {"race_id": race_id, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "file not in data/raw"}
    results = fetch_race_results(season, round_num)
    if not results:
        return {"race_id": race_id, "laps_compared": 0, "laptime_match_share": 0.0, "pit_lap_match_share": 0.0, "driver_laps_match": 0.0, "note": "no reference"}
    driver_map = {}
    expected_laps = {}
    for r in results:
        code = r.get("Driver", {}).get("code")
        did = r.get("Driver", {}).get("driverId")
        if code and did:
            driver_map[did] = code
        if code:
            expected_laps[code] = int(r.get("laps", 0))
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
            did = timing.get("driverId")
            code = driver_map.get(did)
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
    ref_pits = set()
    for ps in pitstops:
        code = driver_map.get(ps.get("driverId"))
        if code:
            ref_pits.add((code, int(ps.get("lap", 0))))
    raw_pits = set()
    in_laps = raw_df[raw_df["PitInTime"].notna()]
    for _, r in in_laps.iterrows():
        raw_pits.add((r["Driver"], int(r["LapNumber"])))
    if not ref_pits and not raw_pits:
        return 1.0
    common = len(ref_pits.intersection(raw_pits))
    total = max(len(ref_pits), len(raw_pits))
    return common / total if total > 0 else 1.0


def evaluate_driver_laps(raw_df, expected_laps):
    """Check driver completed lap count against official classification."""
    matches = 0
    checked = 0
    for drv, g in raw_df.groupby("Driver"):
        if drv in expected_laps:
            checked += 1
            max_lap = int(g["LapNumber"].max())
            if max_lap == expected_laps[drv]:
                matches += 1
    return matches / checked if checked > 0 else 1.0


def run_all_third_source_checks():
    """Run third source check for all races and output reports/testing/third_source.csv."""
    os.makedirs("reports/testing", exist_ok=True)
    df_races = pd.read_csv("reports/races.csv")
    valid_races = df_races[df_races["status"].isin(["downloaded", "cached"])]
    records = []
    print(f"Starting Jolpica third-source check on {len(valid_races)} valid races...", flush=True)
    for idx, (_, row) in enumerate(valid_races.iterrows()):
        yr = int(row["season"])
        rnd = int(row["round"])
        rid = f"race_{yr}_{rnd:02d}"
        res = compare_race(rid, yr, rnd)
        records.append(res)
        print(f"[{idx+1}/{len(valid_races)}] {rid}: cmp={res['laps_compared']}, lt={res['laptime_match_share']}, pit={res['pit_lap_match_share']}, note={res['note']}", flush=True)
        pd.DataFrame(records).to_csv("reports/testing/third_source.csv", index=False)
    print("Third-source verification complete. Saved to reports/testing/third_source.csv", flush=True)


if __name__ == "__main__":
    run_all_third_source_checks()
