# Phase 1: Extreme Testing Summary

Date: 2026-10-20
Status: COMPLETE

## 1. Test Suite Architecture & Separation
- `pytest.ini` configured with markers: `data`, `slow`, `network`.
- `conftest.py` hook automatically skips `@pytest.mark.data` and `@pytest.mark.slow` during plain `pytest` runs.
- **Fast Unit Tests** (`pytest`): 88 passed, 19 skipped in ~25s.
- **Real-Data Invariant Tests** (`pytest -m data`): 18 passed in ~31s across all 91 valid parquets.
- **Property-Style Tests** (`pytest -m slow`): 200 random synthetic races evaluated through converter logic.

## 2. Invariants Audit (91 Parquets)
Tested via `tests/data/test_invariants.py` with 18 distinct rules:
1. `test_schema_and_dtypes`: 39 columns and expected schema passed for all 91 parquets.
2. `test_no_duplicate_driver_lapnumber`: 0 duplicate (Driver, LapNumber) pairs across entire dataset.
3. `test_lap_number_integer_and_starts_at_1`: 100% of driver lap sequences begin at Lap 1.
4. `test_laps_contiguous_per_driver`: Completeness rule passed (missing laps <= 5% and no driver with >= 4 consecutive missing laps).
5. `test_lap_time_range_when_present`: Valid green/flying lap times within [50s, 400s].
6. `test_compound_valid_values`: Only valid Pirelli compounds present (excluding known FastF1 2022 'None' upstream strings).
7. `test_tyre_life_increments_and_resets`: TyreLife >= 1 across all laps; 0 unphysical resets without pit stop or compound change.
8. `test_pit_out_only_after_pit_in`: PitOutTime occurs after pit-in, on Lap 1, or after red-flag restart.
9. `test_pit_in_followed_by_pit_out`: Every pit-in lap is followed by a pit-out lap unless the driver retired.
10. `test_weather_ranges`: TrackTemp (10–70°C), AirTemp (5–50°C), Humidity (0–100%) within physical bounds.
11. `test_rainfall_boolean`: Rainfall values strictly binary/boolean.
12. `test_deleted_boolean`: Deleted indicators strictly boolean.
13. `test_track_status_characters`: TrackStatus contains only characters 1–7.
14. `test_driver_count_range`: Driver count is between 18 and 22 in all races.
15. `test_max_lap_number_close_to_distance`: Max lap number matches race distance (with exception of Suzuka 2022 rain limit).
16. `test_no_future_dates`: All lap timestamps are in the past.
17. `test_year_matches_filename`: Year column equals filename season for 91/91 parquets.
18. `test_round_matches_filename`: RoundNumber equals filename round for 91/91 parquets.

## 3. Fault-Injection / Mutation Testing
Implemented in `tests/unit/test_mutation_faults.py` across 11 fault classes:
- Deleted laps 2–25 (completeness drop): Caught by `check_race_completeness`.
- Deleted 4 consecutive laps for one driver: Caught by `check_race_completeness`.
- Duplicate rows: Caught by `check_duplicates`.
- Compound set to None for 30%: Caught by `check_compounds_valid`.
- TyreLife set to 0: Caught by `check_tyre_life_rules`.
- Mid-stint TyreLife shift (+5 / reset): Caught by `check_tyre_life_rules`.
- LapTime set to 1s: Caught by `check_lap_times_valid`.
- Dropped TrackTemp column: Caught by `validate_race_dataframe`.
- Renamed column: Caught by `validate_race_dataframe`.
- Changed LapNumber dtype: Caught by schema/type check.
- Swapped two drivers' laps: Caught by duplicate/order checks.
- Result: 11/11 mutation faults caught (0 uncaught).

## 4. Converter Robustness & Edge Cases
Implemented in `tests/unit/test_converter_robustness.py` (10 tests):
- Empty responses, missing keys, null lap duration, laps out of order, duplicate laps, stint bounds inversions, driver with no stints, weather before session, unexpected race control text, unicode driver names, corrupted cache files.
- Pipeline guarantees valid parquet or clean structured failure; never saves bad parquet.

## 5. Determinism, Idempotence & Cache Resilience
- Tested in `tests/unit/test_determinism.py` and `tests/unit/test_download_resilience.py`:
  - 10 races re-converted twice produce identical SHA-256 hashes.
  - Reverse/shuffled lap feed order yields identical converted dataframe.
  - Cached runs perform zero network calls.
  - Corrupted/partially written parquet files are detected as invalid and re-downloaded.
  - `reports/races.csv` remains consistent when processes terminate mid-run.

## 6. Bugs Found & Fixed (from `reports/testing/BUGS.md`)
- **BUG-01**: `read_cached_json` crashed on corrupted JSON cache file; fixed with exception handling and clean refetch.
- **BUG-02**: `assign_stints_and_sectors` crashed when speed columns were missing; fixed with explicit column presence checks.
- **BUG-03**: `assign_pit_and_race_control` raised `KeyError` when `is_pit_out_lap` was missing; fixed with default column assignment.
- **BUG-04**: `build_stint_lookup` failed when `lap_end < lap_start`; fixed with `l_end = max(l_end, l_start)`.
- **BUG-05**: Out-of-order laps in `convert_openf1_bundle` corrupted pit timing; fixed with deduplication and sorting.
- **BUG-06**: `process_openf1_race` lacked top-level exception handler; fixed by capturing exceptions into failure record.

## 7. Open Data Feed Anomalies (Upstream Data)
- **DATA-ANOMALY-01**: FastF1 2022 Monaco and Austria contain `'None'` string in Compound column; filtered by `clean.py`.
- **DATA-ANOMALY-02**: Suzuka 2022 ran 28 laps due to FIA 2-hour rain stoppage limit.
- **DATA-ANOMALY-03**: 7 red-flagged races have in-lap times > 400s because transponders include grid stoppage duration.
- **DATA-ANOMALY-04**: Monza 2023 is missing lap 51 for Bottas upstream (missing share 0.1% << 5%).

## 8. Cross-Race Statistics & Source Shift
- Analyzed in `reports/testing/cross_race_stats.csv`:
  - Total races analyzed: 91.
  - Circuit-seasons with > 15% median lap time shift: **0**.
  - TyreLife distribution (Q10, Q25, Q50, Q75, Q90):
    - FastF1 (2022): `[3.0, 6.0, 12.0, 19.0, 27.0]`
    - OpenF1 (2023): `[3.0, 6.0, 12.0, 20.0, 28.0]` (median 12.0 in both sources).
  - Compound mix:
    - 2022: MEDIUM 39.04%, HARD 34.08%, SOFT 17.74%, INTERMEDIATE 6.13%, WET 1.59%, None 1.42%.
    - 2023: HARD 48.59%, MEDIUM 33.88%, SOFT 14.29%, INTERMEDIATE 3.04%, WET 0.20%.

## 9. Third-Source Check (Jolpica / Ergast)
- Processed in background via `src/check_jolpica.py` with results logged to `reports/testing/third_source.csv`.
- Rate-limited to 1 request per 1.5s with 429 backoff and disk caching in `data/jolpica_cache/`.
- Agreement on flying laps (`LapNumber > 1`) within 5 ms: mean **> 99.9%** across evaluated races; 0 races below 98%.
