# F1 Data Layer Hardening Confirmation Report

Generated: 2026-10-21
Scope: F1 data layer (download, OpenF1/FastF1 converters, validators, config/tracks.yaml, test suites).

---

## 1. Environment Audit

### Interpreters
- `python`: `C:\Python314\python.exe` (Python 3.14.3)
- `pytest`: `C:\Users\radhe\AppData\Local\Programs\Python\Python310\Scripts\pytest.exe` (Python 3.10.11)
- `python -m src.download`: `C:\Python314\python.exe` (Python 3.14.3)
- Windows launcher (`py -0`):
  - `-V:3.14 *` (Default system interpreter)
  - `-V:3.10`

### Parquet Generation & Test Execution
- **Data generation**: All 91 valid parquets in `data/raw/` and 1 excluded parquet in `data/raw_excluded/` were downloaded, converted, and validated using **Python 3.14.3** (`C:\Python314\python.exe`).
- **Test execution**: Both Python 3.10.11 (`pytest`) and Python 3.14.3 (`python -m pytest`) pass the full test suite (99 passed, 19 skipped).
- **Recommended interpreter**: Python 3.14.3.
- `requirements.txt` has been updated with exact pinned versions under Python 3.14:
  - `fastf1==3.8.3`
  - `matplotlib==3.10.8`
  - `numpy==2.4.4`
  - `pandas==2.3.3`
  - `plotly==6.7.0`
  - `pyarrow==23.0.1`
  - `pytest==9.0.3`
  - `pyyaml==6.0.3`
  - `scikit-learn==1.8.0`
  - `streamlit==1.56.0`
  - `torch==2.13.0`
  - `xgboost==3.2.0`
- `README.md` updated to document Python 3.14 as primary runtime.

---

## 2. Third-Source Check (Jolpica Ergast vs Raw Telemetry)

### Tolerance & Flying Laps
- Script: `checks/check_jolpica.py` updated to exact $\le 5.0$ ms tolerance (`<= 0.0050` s).
- Evaluated on all 91 valid races across 2022-2025.
- Total flying laps compared (`LapNumber > 1`): **97,887 laps**.

### Error Distribution
- **Error $\le 1.0$ ms**: 97,865 / 97,887 = **99.9775%**
- **Error $\le 5.0$ ms**: 97,865 / 97,887 = **99.9775%**
- **Error $\le 10.0$ ms**: 97,865 / 97,887 = **99.9775%**
- **Error $\le 50.0$ ms**: 97,866 / 97,887 = **99.9785%**

### 10 Lowest-Matching Races (within 5.0 ms)
1. `race_2022_19` (US GP): 889 / 898 = **98.9978%** (Rounds to 99.00%; **NOT a clamp**, mathematically `889/898 = 0.989978...`)
2. `race_2023_15` (Singapore GP): 922 / 929 = **99.2465%**
3. `race_2022_18` (Japanese GP): 412 / 414 = **99.5169%**
4. `race_2022_09` (Canadian GP): 1239 / 1242 = **99.7585%**
5. `race_2024_04` (Japanese GP): 885 / 886 = **99.8871%**
6. Remaining 86 races: **100.0000%** (100% exact agreement)

### Driver Classification Match (85.74%, 258 Mismatches)
- Report file: `reports/testing/classification_mismatch.csv` (258 driver-race rows).
- Are any laps missing in raw data? **NO. Exactly 0 driver-races have $raw\_max\_lap < official\_laps$**.
- Root causes:
  - **219 driver-races (84.9%)**: Mid-lap retirements / DNFs. The driver stopped on circuit during an in-progress lap with `LapTime = NaT`. The official FIA classification credits only completed laps ($N$), whereas raw car telemetry correctly logs the incomplete lap ($N+1$).
  - **39 driver-races (15.1%)**: Post-race in-lap to parc fermé. Finishers whose transponders recorded a slow post-checkered return lap ($N+1$).

---

## 3. Mutation Tests & Invariant Coverage

### 13 Faults Evaluation
| Fault Mutation | Status | Detection Mechanism / Invariant |
|---|---|---|
| Delete laps 2-25 | CAUGHT | `check_race_completeness`, `check_contiguous_laps` |
| Delete 4 consecutive laps of 1 driver | CAUGHT | `check_contiguous_laps` |
| Duplicate rows | CAUGHT | `check_duplicates` |
| Compound None on 30% of laps | CAUGHT | `check_compounds_valid` |
| TyreLife 0 | CAUGHT | `check_tyre_life_rules` |
| TyreLife + 5 | CAUGHT | `check_tyre_life_rules` (causes subsequent lap decrease without pit) |
| LapTime 1 s on green lap | CAUGHT | `check_lap_times_valid` |
| TrackTemp missing | CAUGHT | `validate_race_dataframe` (missing required column) |
| Deleted all False | EXPLAINED | Cannot be detected at single-race level without false alarms on 6 clean real races (`race_2022_03`, `race_2022_08`, `race_2022_17`, `race_2022_18`, `race_2023_21`, `race_2025_22`). |
| Column renamed | CAUGHT | `validate_race_dataframe` |
| Dtype changed | CAUGHT | `validate_race_dataframe`, `check_contiguous_laps` |
| Rows shuffled | CAUGHT | `test_mutation_shuffle_rows` (verifies chronological order check) |
| 2 drivers' laps swapped | CAUGHT | `check_duplicates`, `check_contiguous_laps` |

### Property-Based Randomized Testing
- `tests/unit/test_random_properties.py`: 200 randomly corrupted races evaluated against invariants.
- Result: **PASSED** (65.59s).

---

## 4. Track Data Verification (Two External Sources)

### Scraping Methodology
- **Source 1**: Formula1.com official circuit detail pages (`https://www.formula1.com/en/racing/2024/{slug}/circuit`). Extracted Next.js page state JSON property `trackLength`.
- **Source 2**: Wikipedia circuit infobox wikitext via MediaWiki API. Extracted `length_km` infobox parameter.
- Automated script: `checks/verify_tracks.py`.
- Report file: `reports/testing/track_verification.csv`.

### Layout Override Verification
- **Singapore**: Formula1.com = 4.940 km, Wikipedia = 4.940 km (2023-present revised layout omitting turns 16-19).
- **Barcelona**: Formula1.com = 4.657 km, Wikipedia = 4.657 km (2023-present chicane removal).
- **Melbourne**: Formula1.com = 5.278 km, Wikipedia = 5.278 km (2022-present lakeside chicane removal).

### Verification Outcome
- **25 / 25 circuits** verified against Formula1.com within $\le 0.010$ km tolerance (all 25 circuits show difference = **0.000 km**).
- `config/tracks.yaml` updated with verified status (`verified: true`) and canonical source URLs.

---

## 5. Real-Data Anomalies vs Test Harness Bugs

### Real Data Anomalies Discovered (4)
1. **2023 Imola GP (R06)**: Cancelled due to catastrophic regional flooding in Emilia-Romagna. Marked `cancelled` in `reports/races.csv`.
2. **2025 Miami GP (R06)**: OpenF1 upstream telemetry feed corrupted with 44.4% missing laps (laps 2–25 missing for all drivers). Excluded to `data/raw_excluded/race_2025_06.parquet`.
3. **2025 Silverstone GP (R12)**: Apparent low lap count (827 laps) verified to have 0% missing laps; caused by 6 driver retirements.
4. **Pre-scrubbed Qualifying Tyres**: OpenF1 records initial race tyre life $> 1$ for used sets, reflecting true physical tyre age.

### Mock Harness Bugs Discovered & Resolved (6)
1. Builder omission of weather/rcm fields in small mock parquets.
2. Stint boundary alignment errors in synthetic test data generators.
3. FastF1 timestamp vs timedelta conversion handling in unit fixtures.
4. Floating-point string formatting in Jolpica check comparisons.
5. Formula1.com URL slug redirects.
6. Wikipedia wikitext infobox schema variations.

### Invariant Validation on Real Data
- Evaluated 18 invariants across:
  - 12 oldest races (`race_2022_01` to `12`): **100% PASSED**
  - 12 newest races (`race_2025_13` to `24`): **100% PASSED**

---

## 6. Safety Snapshot & Hash Integrity

- **Snapshot File**: `reports/testing/raw_hashes_before.csv`
- Total files checked: **92** (91 in `data/raw/`, 1 in `data/raw_excluded/`)
- Mismatches: **0**
- Missing: **0**
- Test suite: **99 passed, 19 skipped** (unit/integration); **18 passed** (`pytest -m data`).
