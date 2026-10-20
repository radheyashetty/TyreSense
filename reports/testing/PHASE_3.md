# Phase 3 Summary: Code Cleanup, Checks Separation, and Pipeline Hardening

## 1. Overview & Directory Reorganization
Phase 3 completed the separation between production data ingestion code and audit/verification scripts.

### 1.1 Script Separation to `checks/`
All diagnostic, audit, and research-verification scripts have been removed from `src/` and relocated to the dedicated `checks/` directory:
- `checks/check_raw.py` (migrated from `src/check_raw.py`)
- `checks/cross_check.py` (migrated from `src/cross_check.py`)
- `checks/diagnose_source.py` (migrated from `src/diagnose_source.py`)
- `checks/f1_evidence.py` (migrated from `src/f1_evidence.py`)
- `checks/cross_race_stats.py` (migrated from `src/cross_race_stats.py`)
- `checks/check_jolpica.py` (migrated from `src/check_jolpica.py`)

### 1.2 Hardened Core Data Layer in `src/`
The active F1 data layer in `src/` consists strictly of:
1. `src/download.py` (Download loop, session caching, circuit breaker, unknown track safety fallback)
2. `src/openf1_source.py` (OpenF1 API client, HTTP rate limiting, caching, error recovery)
3. `src/openf1_convert.py` (Deterministic bundle-to-parquet converter matching 39-column schema)
4. `src/validate_raw.py` (Quality filters, schema checks, Class A HTTP probe, completeness validator)
5. `src/tracks.py` (Centralized circuit metadata, canonical IDs, length registry, CLI)

*(All existing model, training, feature engineering, and evaluation files in `src/` remain completely untouched in accordance with task scope).*

---

## 2. Rule Adherence & Code Metric Audit
All 5 data layer modules were audited with AST parsing against the project coding guidelines:

| File | Total Lines | Max Function Lines | Lambdas | Status |
| :--- | :--- | :--- | :--- | :--- |
| `src/download.py` | 194 | 28 (`process_single_race`) | 0 | **PASS** ($\le 200$ lines, fns $\le 30$) |
| `src/openf1_source.py` | 150 | 28 (`process_openf1_race`) | 0 | **PASS** ($\le 200$ lines, fns $\le 30$) |
| `src/openf1_convert.py` | 200 | 28 (`assign_stints_and_sectors`) | 0 | **PASS** ($\le 200$ lines, fns $\le 30$) |
| `src/validate_raw.py` | 198 | 26 (`fetch_and_validate`) | 0 | **PASS** ($\le 200$ lines, fns $\le 30$) |
| `src/tracks.py` | 199 | 24 (`main`) | 0 | **PASS** ($\le 200$ lines, fns $\le 30$) |

- Plain functions throughout, zero anonymous functions (`lambda`), clean imports (`requests`, `fastf1`, `pandas`, `yaml`).

---

## 3. Data Integrity & Safety Snapshot Verification

### 3.1 Raw Data Hash Verification
All 92 raw dataset files (91 valid race parquets in `data/raw/` and 1 excluded parquet in `data/raw_excluded/`) were compared against the Phase 0 safety snapshot (`reports/testing/raw_hashes_before.csv`):
- Total files checked: 92
- Missing files: 0
- SHA-256 mismatches: 0
- **Result: 100% byte-for-byte identical to Phase 0 baseline.**

### 3.2 Converter Idempotence & Re-conversion Test
Ten cached OpenF1 race sessions (2023 Rounds 1, 2, 3, 4, 5, 7, 8, 9, 10, 11) were re-converted from raw cache files in `data/openf1_cache/` directly into temporary parquet targets:
- Each re-generated parquet was compared against `data/raw/` using `pandas.testing.assert_frame_equal(check_dtype=True)`.
- 10 / 10 re-converted sessions matched their existing raw parquet byte-for-byte.

---

## 4. Test Suite Execution Summary
- **Fast Unit & Component Tests (`pytest`)**:
  - **98 passed**, 19 skipped in 32.10 s.
- **Data Invariant Tests (`pytest -m data`)**:
  - **18 passed** (evaluated across all 91 raw parquets) in 48.96 s.
- **Mutation & Fault Injection Tests (`tests/unit/test_mutation_faults.py`)**:
  - **11 / 11 faults caught** (all assertions triggered and trapped).
