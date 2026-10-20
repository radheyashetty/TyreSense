# TyreSense Data Layer Bug Tracking (Phase 1)

Date: 2026-10-20

## 1. Bugs Found & Fixed

### BUG-01: `read_cached_json` crashed on corrupted cache file
- **What**: When a local JSON cache file was corrupted, truncated, or invalid, `json.load(f)` raised `json.JSONDecodeError` and crashed the download process.
- **Evidence**: `test_cache_corrupted_json_returns_none` in `tests/unit/test_converter_robustness.py`.
- **Fix**: Wrapped file loading in `try...except Exception: return None` so corrupted cache entries are cleanly refetched from network.
- **Test name**: `tests/unit/test_converter_robustness.py::test_cache_corrupted_json_returns_none`.

### BUG-02: `assign_stints_and_sectors` crashed on missing speed columns
- **What**: When optional speed fields (e.g. `i1_speed`, `st_speed`) were missing from raw JSON, `df.get("i1_speed", np.nan).astype(float)` raised `AttributeError: 'float' object has no attribute 'astype'`.
- **Evidence**: `AttributeError` during `test_converter_missing_keys`.
- **Fix**: Replaced scalar `.get()` with explicit column presence checks: `df["i1_speed"].astype(float) if "i1_speed" in df.columns else np.nan`.
- **Test name**: `tests/unit/test_converter_robustness.py::test_converter_missing_keys`.

### BUG-03: `assign_pit_and_race_control` crashed on missing `is_pit_out_lap`
- **What**: If OpenF1 laps endpoint omitted `is_pit_out_lap`, accessing `grp["is_pit_out_lap"]` raised `KeyError`.
- **Evidence**: `KeyError: 'is_pit_out_lap'` in `tests/unit/test_converter_robustness.py`.
- **Fix**: Added default column initialisation: `if "is_pit_out_lap" not in df.columns: df["is_pit_out_lap"] = False`.
- **Test name**: `tests/unit/test_converter_robustness.py::test_converter_missing_keys`.

### BUG-04: `build_stint_lookup` failed when `lap_end < lap_start`
- **What**: Corrupted or out-of-order stint end bounds caused empty range iteration and unassigned stint metadata.
- **Evidence**: `test_converter_stint_end_before_start_and_overlapping`.
- **Fix**: Clamped `l_end = max(l_end, l_start)`.
- **Test name**: `tests/unit/test_converter_robustness.py::test_converter_stint_end_before_start_and_overlapping`.

### BUG-05: `convert_openf1_bundle` disordered laps corrupted pit in/out indexing
- **What**: If OpenF1 returned laps out of chronological order or with duplicates, relative index lookups in `assign_pit_and_race_control` produced invalid pit timings.
- **Evidence**: `test_converter_laps_out_of_order_and_duplicates`.
- **Fix**: Deduplicated by `(driver_number, lap_number)` and sorted before assigning pit and stint timings.
- **Test name**: `tests/unit/test_converter_robustness.py::test_converter_laps_out_of_order_and_duplicates`.

### BUG-06: `process_openf1_race` lacked top-level exception handler
- **What**: Network timeouts, HTTP 500 errors, or connection resets during fetch caused an unhandled crash before logging race status.
- **Evidence**: `test_process_openf1_race_handles_errors_gracefully`.
- **Fix**: Wrapped `fetch_and_convert_openf1_race` in `try...except Exception as err:` returning structured failure status dict.
- **Test name**: `tests/unit/test_converter_robustness.py::test_process_openf1_race_handles_errors_gracefully`.

---

## 2. Open Data Feed Anomalies (Upstream Data Integrity)

1. **DATA-ANOMALY-01 (`race_2022_07`, `race_2022_11`)**: FastF1 upstream timing records compound as `'None'` string when tyre compound was not recorded (Stroll in Monaco, Gasly in Austria). Handled downstream by `clean.py` compound filter.
2. **DATA-ANOMALY-02 (`race_2022_18`)**: Suzuka 2022 max lap is 28 laps due to FIA 2-hour rain stoppage limit.
3. **DATA-ANOMALY-03 (Red Flag Stoppages)**: 7 races (`race_2023_03`, `race_2023_14`, `race_2023_17`, `race_2023_20`, `race_2023_21`, `race_2024_04`, `race_2024_21`) have recorded in-lap times > 400 s because official F1 timing transponders include race suspension time in pit lane. Handled by `clean.py` disruption filtering.
4. **DATA-ANOMALY-04 (`race_2023_15`)**: Monza 2023 is missing lap 51 for Bottas upstream (missing rate 0.1% << 5% threshold).
