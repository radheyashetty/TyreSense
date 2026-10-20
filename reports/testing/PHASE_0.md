# Phase 0: Safety Snapshot Summary

Date: 2026-10-20
Status: COMPLETE

## 1. File Hashing
- Computed SHA-256 for all parquet files in `data/raw/` (91 files) and `data/raw_excluded/` (1 file: `race_2025_06.parquet`).
- Total files hashed: 92.
- Hash registry saved to: `reports/testing/raw_hashes_before.csv`.

## 2. Races Status Backup
- Copied `reports/races.csv` to `reports/testing/races_before.csv`.
- Validated backup exists and matches original size.

## 3. Baseline Pytest Run
- Command: `python -m pytest`
- Result: 61 passed in 31.77s (100% pass rate).
