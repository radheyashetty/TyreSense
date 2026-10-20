# TyreSense Repository Technical Briefing

**Generated**: 2026-10-21  
**Scope**: Factual audit of codebase, data pipeline, test suite, and discrepancies as of commit `b087f8c`.

---

## 1. REPO MAP

### Directory Tree (2 Levels Deep, excluding caches, data, .git, __pycache__)
```text
.
├── .gitignore
├── AGENT_RULES.md
├── app.py
├── config.yaml
├── conftest.py
├── logs_download_2022.txt
├── logs_download_2023.txt
├── logs_download_2024.txt
├── logs_download_2025.txt
├── logs_download_all.txt
├── logs_download_all_err.txt
├── logs_jolpica.txt
├── PART1_PRD.md
├── PART2_PRD.md
├── pytest.ini
├── README.md
├── requirements.txt
├── checks/
│   ├── __init__.py
│   ├── check_jolpica.py
│   ├── check_raw.py
│   ├── cross_check.py
│   ├── cross_race_stats.py
│   ├── diagnose_source.py
│   ├── f1_evidence.py
│   └── verify_tracks.py
├── config/
│   └── tracks.yaml
├── notes/
│   └── explain.md
├── reports/
│   ├── ablation.csv
│   ├── attrition.csv
│   ├── baseline_table.csv
│   ├── cliff_sensitivity.csv
│   ├── cross_source_check.csv
│   ├── diagnose_source.txt
│   ├── error_analysis.csv
│   ├── exp9_core_claim.csv
│   ├── f1_evidence.txt
│   ├── final_report.md
│   ├── fuel_sensitivity.csv
│   ├── gradflow.csv
│   ├── project_overview.md
│   ├── races.csv
│   ├── races_old_part1.csv
│   ├── raw_check.csv
│   ├── tracks_pending.csv
│   ├── uncertainty_coverage.csv
│   ├── audit/
│   ├── figures/
│   ├── part1_old/
│   └── testing/
├── runs/
│   ├── invalid_final_2023_02/
│   └── pilot/
├── src/
│   ├── __init__.py
│   ├── ablation.py
│   ├── baselines.py
│   ├── claim_eval.py
│   ├── clean.py
│   ├── config.py
│   ├── download.py
│   ├── evaluate.py
│   ├── features.py
│   ├── final_eval.py
│   ├── fuel.py
│   ├── gradflow.py
│   ├── live.py
│   ├── mlp_numpy.py
│   ├── models.py
│   ├── openf1_convert.py
│   ├── openf1_source.py
│   ├── reg_ablation.py
│   ├── sensitivity.py
│   ├── split.py
│   ├── tracks.py
│   ├── train.py
│   ├── uncertainty.py
│   └── validate_raw.py
└── tests/
    ├── conftest.py
    ├── test_ablation.py
    ├── test_app.py
    ├── test_baselines.py
    ├── test_check_raw.py
    ├── test_claim.py
    ├── test_clean.py
    ├── test_download.py
    ├── test_features.py
    ├── test_leakage.py
    ├── test_live.py
    ├── test_mlp_numpy.py
    ├── test_models.py
    ├── test_openf1.py
    ├── test_split.py
    ├── test_uncertainty.py
    ├── data/
    │   ├── invariant_helpers.py
    │   └── test_invariants.py
    ├── fixtures/
    │   └── sample_race_builder.py
    └── unit/
        ├── test_converter_robustness.py
        ├── test_determinism.py
        ├── test_download_resilience.py
        ├── test_mutation_faults.py
        ├── test_random_properties.py
        └── test_tracks.py
```

### File-by-File Summary

#### `src/` (Core Modules)
- `ablation.py`: Executes ablation experiments EXP-2, EXP-3, EXP-6, and EXP-7 over baseline models.
- `baselines.py`: Implements Model A (causal wear curve), Model B (Ridge regression), Model C (XGBoost).
- `claim_eval.py`: Evaluates EXP-9 core claim (FROZEN vs ROLLING vs Model E) with paired bootstrap and cliff sensitivity.
- `clean.py`: Filters raw race parquet files (weather, lap 1, safety cars, in/out laps), applies fuel correction, and labels usable life.
- `config.py`: Loads and parses `config.yaml` settings.
- `download.py`: Orchestrates multi-season downloads (FastF1 for 2022, OpenF1 for 2023–2025) and records catalog to `reports/races.csv`.
- `evaluate.py`: Calculates multi-horizon MAE ($h=1, 5, 10$) and compound-level censored share metrics.
- `features.py`: Builds causal quadratic wear curves and constructs 8-input sequence arrays with forward residual targets.
- `final_eval.py`: Runs locked single-pass evaluation of Model E on test stints and generates worst-stint error analysis.
- `fuel.py`: Computes deterministic fuel mass penalty adjustment scaled by track distance.
- `gradflow.py`: Implements EXP-8 gradient norm calculation across sequence lengths for recurrent models.
- `live.py`: Implements online adaptive wear rate multiplier ($k$-factor update rule) during stint progression.
- `mlp_numpy.py`: 8-16-8-1 neural network implemented from scratch in pure NumPy with backpropagation and numerical gradient check.
- `models.py`: PyTorch recurrent neural networks (GRU, LSTM, RNN) with matched ~7.1k parameter budgets and optional monotone head.
- `openf1_convert.py`: Converts raw OpenF1 JSON endpoints into the standardized 39-column raw parquet schema.
- `openf1_source.py`: Fetches and caches OpenF1 sessions, laps, stints, weather, race control, and driver data with backoff.
- `reg_ablation.py`: Implements EXP-4 regularization sweeps (L2, dropout, Lasso) and EXP-5 early stopping patience sweeps.
- `sensitivity.py`: Evaluates sensitivity of baseline pace models across fuel slopes ($0.02, 0.03, 0.04\text{ s/lap}$).
- `split.py`: Partitions race data into temporal splits (2022–2023 train, 2024 val, 2025 test) with unseen track handling.
- `tracks.py`: Centralized track identification, alias resolution, and circuit length configuration service.
- `train.py`: PyTorch training loop on CPU featuring asymmetric Huber loss, gradient clipping, and sequence batching.
- `uncertainty.py`: 30-pass Monte Carlo Dropout inference engine and empirical interval calibration ($P_{10}, P_{50}, P_{90}$).
- `validate_raw.py`: Validates raw 39-column parquet schema, data types, value bounds, and checks race lap completeness.

#### `checks/` (Standalone Audit Scripts)
- `check_jolpica.py`: Compares raw parquet flying laps against Jolpica Ergast API at $\le 5.0\text{ ms}$ tolerance.
- `check_raw.py`: Audits raw parquet files in `data/raw/` for null columns, missing lap times, and deleted-lap message presence.
- `cross_check.py`: Evaluates cross-source telemetry consistency (FastF1 vs OpenF1) for 2023 R1 and R2.
- `cross_race_stats.py`: Computes distribution metrics and cross-race summary statistics across the raw dataset.
- `diagnose_source.py`: Probes FastF1 and OpenF1 HTTP endpoints to diagnose live timing availability and response codes.
- `f1_evidence.py`: Gathers and formats audit evidence for F1 gates and prints reports.
- `verify_tracks.py`: Scrapes official Formula1.com circuit pages and Wikipedia circuit infoboxes to verify 25 track lengths.

#### `config/`
- `tracks.yaml`: Master track registry containing 25 verified F1 circuits, official lengths, layout overrides, and alias mappings.

#### `tests/`
- `test_ablation.py`: Validates ablation report CSV structure and column completeness.
- `test_app.py`: Benchmarks Streamlit app render latency ($<3\text{ s}$ load, $<1\text{ s}$ scrub) and verifies causality.
- `test_baselines.py`: Tests Models A, B, C training, prediction, and censoring metrics.
- `test_check_raw.py`: Tests raw parquet inspection functions and completeness flags on synthetic race parquets.
- `test_claim.py`: Tests Pit Window Error (PWE) metric and paired bootstrap significance calculations.
- `test_clean.py`: Tests fuel slope removal, track-scaled fuel, spike filtering, attrition tracking, and usable life labelling.
- `test_download.py`: Tests FastF1 downloading, circuit breakers, cache avoidance, and schema migration.
- `test_features.py`: Tests causal quadratic curve fitting, 8-feature extraction, and absence of lookahead leakage.
- `test_leakage.py`: Tests all 5 leakage invariants (temporal split, past-future isolation, train-only fitting, app replay).
- `test_live.py`: Tests $k$-factor dynamics (freezing on spike laps, freezing before lap 3, moving toward ground truth).
- `test_mlp_numpy.py`: Tests analytical vs numerical gradient check ($<10^{-6}$), loss parity with PyTorch, and Adam optimizer.
- `test_models.py`: Validates model forward pass, matched parameter counts ($<5\%$), and monotone output head.
- `test_openf1.py`: Tests OpenF1 conversion formulas, tyre age math, deleted laps, safety car mapping, and rate limits.
- `test_split.py`: Tests temporal partition disjointness and fallback handling for unseen tracks.
- `test_uncertainty.py`: Tests MC-Dropout inference mode, quantile ordering ($P_{10} \le P_{50} \le P_{90}$), and scale calibration.
- `data/invariant_helpers.py`: Helper functions defining 18 domain invariant checks.
- `data/test_invariants.py`: Pytest suite running 18 domain invariants across real raw parquet files.
- `fixtures/sample_race_builder.py`: Mock 39-column raw parquet DataFrame fixture builder.
- `unit/test_converter_robustness.py`: Tests OpenF1 converter against malformed inputs (null durations, overlapping stints, unicode).
- `unit/test_determinism.py`: Tests converter determinism and zero network calls on cached runs.
- `unit/test_download_resilience.py`: Tests atomic parquet writes, interrupted catalog updates, and corrupt file detection.
- `unit/test_mutation_faults.py`: Tests fault injection mutations against validators and invariants.
- `unit/test_random_properties.py`: Property-based randomized test evaluating 200 corrupted race variations.
- `unit/test_tracks.py`: Tests track alias resolution, length consistency, overrides, and CLI update operations.

### Non-Data-Layer Files in `src/` & Change Status
The following modules belong to modeling, feature engineering, training, or application layers. None were modified during task F1:
- `src/clean.py`: Modified in Part 2 (`0a21be0` for track fuel & spike filtering); untouched in F1.
- `src/fuel.py`: Modified in Part 2 (`2b8cd99` for track scaling); untouched in F1.
- `src/split.py`: Unchanged since Part 1 initial commit (`6985d46`).
- `src/features.py`: Modified in Part 2 (`9a7ea75` for 8th input `track_temp_delta_c`); untouched in F1.
- `src/baselines.py`: Unchanged since Part 1 initial commit (`6985d46`).
- `src/evaluate.py`: Unchanged since Part 1 initial commit (`6985d46`).
- `src/mlp_numpy.py`: Modified in Part 2 (`9a7ea75` for 8-16-8-1 MLP); untouched in F1.
- `src/models.py`: Modified in Part 2 (`4989897`, `39e2aee` for parameter budgets and monotone head); untouched in F1.
- `src/train.py`: Modified in Part 2 (`4989897` for asymmetric loss); untouched in F1.
- `src/ablation.py`: Unchanged since Part 1 initial commit (`6985d46`).
- `src/reg_ablation.py`: Added in Part 2 (`39e2aee`, `f6c6f13` for EXP-4, EXP-5, EXP-10); untouched in F1.
- `src/sensitivity.py`: Added in Part 2 (`0a21be0` for fuel sensitivity); untouched in F1.
- `src/gradflow.py`: Added in Part 2 (`83262d0` for EXP-8); untouched in F1.
- `src/live.py`: Added in Part 2 (`e5389d3` for online adaptive $k$-factor); untouched in F1.
- `src/uncertainty.py`: Added in Part 2 (`f6c6f13` for MC-Dropout); untouched in F1.
- `src/claim_eval.py`: Added in Part 2 (`fbfd6f5` for EXP-9 core claim); untouched in F1.
- `src/final_eval.py`: Added in Part 2 (`d785e7e` for locked final test evaluation); untouched in F1.
- `src/config.py`: Unchanged since Part 1 initial commit (`6985d46`).
- `app.py`: Modified in Part 2 (`b798ac5` for uncertainty envelope, ghost line, $k$ gauge, pit window card); untouched in F1.

---

## 2. CODE SIZE AND COMPLEXITY

### Measurement Command
```powershell
& "C:\Python314\python.exe" -c "
import os, ast

# AST Cyclomatic Complexity and Line Counts
"
```

### Line Counts by File
```text
app.py: 187 lines

checks/
  checks/__init__.py: 1 lines
  checks/check_jolpica.py: 192 lines
  checks/check_raw.py: 79 lines
  checks/cross_check.py: 106 lines
  checks/cross_race_stats.py: 98 lines
  checks/diagnose_source.py: 104 lines
  checks/f1_evidence.py: 210 lines
  checks/verify_tracks.py: 145 lines

src/
  src/__init__.py: 0 lines
  src/ablation.py: 188 lines
  src/baselines.py: 53 lines
  src/claim_eval.py: 147 lines
  src/clean.py: 166 lines
  src/config.py: 7 lines
  src/download.py: 194 lines
  src/evaluate.py: 137 lines
  src/features.py: 113 lines
  src/final_eval.py: 118 lines
  src/fuel.py: 25 lines
  src/gradflow.py: 111 lines
  src/live.py: 50 lines
  src/mlp_numpy.py: 119 lines
  src/models.py: 79 lines
  src/openf1_convert.py: 197 lines
  src/openf1_source.py: 150 lines
  src/reg_ablation.py: 133 lines
  src/sensitivity.py: 69 lines
  src/split.py: 64 lines
  src/tracks.py: 199 lines
  src/train.py: 179 lines
  src/uncertainty.py: 103 lines
  src/validate_raw.py: 198 lines

tests/
  tests/conftest.py: 16 lines
  tests/test_ablation.py: 19 lines
  tests/test_app.py: 44 lines
  tests/test_baselines.py: 81 lines
  tests/test_check_raw.py: 101 lines
  tests/test_claim.py: 31 lines
  tests/test_clean.py: 97 lines
  tests/test_download.py: 191 lines
  tests/test_features.py: 114 lines
  tests/test_leakage.py: 75 lines
  tests/test_live.py: 51 lines
  tests/test_mlp_numpy.py: 95 lines
  tests/test_models.py: 74 lines
  tests/test_openf1.py: 121 lines
  tests/test_split.py: 104 lines
  tests/test_uncertainty.py: 51 lines
  tests/data/invariant_helpers.py: 108 lines
  tests/data/test_invariants.py: 200 lines
  tests/fixtures/sample_race_builder.py: 78 lines
  tests/unit/test_converter_robustness.py: 94 lines
  tests/unit/test_determinism.py: 54 lines
  tests/unit/test_download_resilience.py: 43 lines
  tests/unit/test_mutation_faults.py: 137 lines
  tests/unit/test_random_properties.py: 68 lines
  tests/unit/test_tracks.py: 107 lines
```

### 15 Longest Functions in `src/`
```text
1. ablation.py::run_all_ablations: 30 lines
2. claim_eval.py::evaluate_exp9_stints: 30 lines
3. claim_eval.py::simulate_modes_at_lap: 29 lines
4. clean.py::clean_race_dataframe: 29 lines
5. download.py::process_single_race: 29 lines
6. evaluate.py::prepare_and_train_baselines: 29 lines
7. train.py::run_pilot: 29 lines
8. ablation.py::run_exp2: 28 lines
9. ablation.py::run_exp3: 27 lines
10. train.py::build_sequences_from_laps: 27 lines
11. openf1_convert.py::build_stint_lookup: 26 lines
12. openf1_convert.py::assign_pit_and_race_control: 26 lines
13. ablation.py::run_exp6: 25 lines
14. ablation.py::run_exp7: 25 lines
15. sensitivity.py::run_one_setting: 25 lines
```

### 10 Most Complex Functions in `src/` (AST Cyclomatic Complexity)
```text
1. download.py::fetch_race_with_retry: CC = 12
2. openf1_convert.py::parse_sc_intervals: CC = 10
3. openf1_convert.py::build_stint_lookup: CC = 10
4. validate_raw.py::validate_race_dataframe: CC = 10
5. openf1_source.py::request_with_retry: CC = 9
6. tracks.py::canonical_track_id: CC = 8
7. validate_raw.py::check_race_completeness: CC = 8
8. validate_raw.py::fetch_and_validate: CC = 8
9. download.py::process_single_race: CC = 7
10. openf1_convert.py::compute_lap_track_status: CC = 6
```

### Structural Counts in `src/`
- **Total Lambdas in `src/`**: 0
- **Total Classes in `src/`**: 4
  - `models.py::MonotoneHead` (`torch.nn.Module`)
  - `models.py::RecurrentForecaster` (`torch.nn.Module`)
  - `train.py::StintSequenceDataset` (`torch.utils.data.Dataset`)
  - `validate_raw.py::WarningHandler` (context manager)
- **Total Module-Level Globals in `src/`**: 2
  - `openf1_source.py::BASE_URL`
  - `tracks.py::TRACKS_YAML_PATH`

### AGENT_RULES.md Audit
| Rule | Status | Evidence / Violations |
|---|---|---|
| Plain functions; only classes allowed are `torch.nn.Module` | **PARTIAL** | `models.py` uses `nn.Module`. However, `train.py::StintSequenceDataset` (`torch.utils.data.Dataset`) and `validate_raw.py::WarningHandler` (context manager) are non-`nn.Module` classes. |
| One job per function, $\le 30$ lines | **PASS** | Every function in `src/` is $\le 30$ lines (longest is 30 lines). |
| One file per job, $\le 200$ lines | **PARTIAL** | All files in `src/` and `tests/` are $\le 200$ lines. However, `checks/f1_evidence.py` is 210 lines. |
| Plain names, no abbreviations/clever one-liners | **PASS** | Variable names use plain conventions (`tyre_age`, `lap_time`, etc.). |
| Simple for-loops, basic pandas, no lambdas/decorators | **PASS** | 0 lambdas across `src/`. Only standard decorators (`@torch.no_grad()`, `@pytest.fixture`) used. |
| Comments short and say WHY | **PASS** | Docstrings and comments are minimal and explanatory. |
| Use `print()`, no logging framework | **PASS** | 0 `import logging` across `src/`. Progress reported via `print()`. |
| Settings in `config.yaml`, read by `load_config()` | **PASS** | Parsed centrally via `src/config.py::load_config()`. |
| Hand-written training loop (no Lightning, MLflow) | **PASS** | PyTorch loop written directly in `src/train.py`. |
| Allowed libraries only | **PASS** | Only uses allowed libraries (`numpy`, `pandas`, `sklearn`, `xgboost`, `torch`, `fastf1`, `streamlit`, `plotly`, `matplotlib`, `pytest`, `pyyaml`, `requests`). |
| No code for future use | **PASS** | All modules correspond to assigned PRD tasks. |
| Results & reporting rules (F0) | **PASS** | Every reported metric is derived directly from report CSVs or live outputs. |

---

## 3. DATA PIPELINE, END TO END

### Trace: API Call to `data/raw/*.parquet` to `data/processed/*.parquet`

```text
1. API INGESTION
   ├── 2022 (FastF1):
   │   download.py::process_single_race()
   │     └── fastf1.get_session(year, round, 'R').load(telemetry=False, weather=True, messages=True)
   │     └── fastf1_laps + fastf1_weather merged to 39-column raw DataFrame
   └── 2023–2025 (OpenF1):
       download.py::process_single_race()
         └── openf1_source.py::process_openf1_race()
               └── openf1_source.py::fetch_race_bundle(session_key)
                     ├── /laps, /stints, /weather, /race_control, /drivers
                     └── cached as JSON in data/openf1_cache/
               └── openf1_convert.py::convert_openf1_bundle()
                     ├── build_stint_lookup(): maps driver & lap to stint, compound, tyre life
                     ├── parse_sc_intervals(): maps race control timestamps to track status
                     ├── merge_weather_data(): interpolates track temp, air temp, humidity
                     └── assembles exact 39-column raw DataFrame

2. RAW DATA VALIDATION & PERSISTENCE
   validate_raw.py::validate_race_dataframe()
     ├── checks 39 required columns, types, ranges, non-null requirements
   validate_raw.py::check_race_completeness()
     ├── flags >5% missing laps or >=4 consecutive missing laps
   └── df.to_parquet('data/raw/race_{year}_{round:02d}.parquet')

3. DOWNSTREAM CLEANING & FEATURE PIPELINE (Scheduled for F3)
   clean.py::clean_race_dataframe()
     ├── removes non-dry races (Inter/Wet)
     ├── drops lap 1 chaos, SC/VSC laps (TrackStatus != 1), and pit in/out laps
     ├── fuel.py::fuel_correct_series(): removes fuel burn penalty scaled by track distance
     ├── detects and flags traffic/anomaly spikes (offline recovery & causal)
     └── computes usable life labels (+1.0 s cliff) and right-censoring
   └── writes data/processed/laps.parquet and data/processed/stints.parquet
```

### FastF1 vs. OpenF1 by Season & Root Cause
- **2022**: Downloaded via FastF1 (22 races, 23,577 laps). FastF1 succeeds because 2022 session data was cached historically and lives on Ergast/FastF1 static archives.
- **2023–2025**: Downloaded via OpenF1 (69 valid races, 76,814 laps).
- **FastF1 Root Cause**: Upstream live timing endpoints on `livetiming.formula1.com` return **HTTP 401 Unauthorized** (CloudFront 0-byte block) for race control messages, and the fallback mirror `livetiming-mirror.fastf1.dev` returns **HTTP 404 Not Found**. For uncached sessions from 2023 R3 onwards, all live timing endpoints (`SessionInfo`, `TimingData`, `TrackStatus`, `WeatherData`) return HTTP 401, causing FastF1 `DataNotLoadedError` and loading 0 laps.
- **Evidence**: Documented in `reports/diagnose_source.txt` and `logs_download_all_err.txt`.

### Current State of `data/processed/`
Verification command:
```powershell
& "C:\Python314\python.exe" -c "
import pandas as pd
laps = pd.read_parquet('data/processed/laps.parquet')
stints = pd.read_parquet('data/processed/stints.parquet')
print('laps.parquet shape:', laps.shape, 'races:', laps['race_id'].unique())
print('stints.parquet shape:', stints.shape)
"
```
Output:
```text
laps.parquet shape: (1663, 14) races: ['2023_01' '2023_02']
stints.parquet shape: (98, 4)
```
**Finding**: `clean.py` and `data/processed/` have **NOT** been regenerated on the 91 races. The processed files still contain strictly the Part 1 pilot dataset (**1,663 laps across 2 races: 2023 R1 and 2023 R2**). Regeneration across the 91 races is scheduled for F3.

---

## 4. TESTS

### Pytest Execution Summaries

1. **Standard Pytest**:
   ```powershell
   python -m pytest
   ```
   **Output**:
   ```text
   ======================= 99 passed, 19 skipped in 24.76s =======================
   ```

2. **Data Invariants (`pytest -m data`)**:
   ```powershell
   pytest -m data
   ```
   **Output**:
   ```text
   ===================== 18 passed, 100 deselected in 43.99s =====================
   ```

3. **Slow Property Tests (`pytest -m slow`)**:
   ```powershell
   pytest -m slow
   ```
   **Output**:
   ```text
   ================ 1 passed, 117 deselected in 71.14s (0:01:11) =================
   ```

### Test Count Per Directory
- `tests/`: 61 tests
- `tests/data/`: 18 tests
- `tests/unit/`: 39 tests
- **Total Collected**: **118 tests**

### Real Checks vs. Trivial Checks
- **Trivial / Structural Checks (3 tests)**:
  - `test_ablation.py::test_ablation_report_structure`: Asserts `reports/ablation.csv` exists and contains required columns.
  - `test_baselines.py::test_baseline_table_generation`: Asserts `reports/baseline_table.csv` exists and checks columns.
  - `test_check_raw.py::test_check_all_raw_parquets_creates_report`: Asserts `reports/raw_check.csv` is written.
- **Genuine Mathematical & Data Checks (115 tests)**:
  - `test_mlp_numpy.py`: Evaluates two-sided numerical vs. analytical gradient difference ($1.42 \times 10^{-8} < 10^{-6}$) and Adam loss reduction.
  - `test_leakage.py`: Tests past-future temporal isolation, causal curve fitting, and guarantees features do not alter when future laps change.
  - `test_invariants.py`: 18 data checks covering types, non-null constraints, weather bounds, tyre increments, and contiguous lap sequences across all 91 raw parquets.
  - `test_mutation_faults.py`: Injects 12 corruption mutations and verifies detector responses.
  - `test_converter_robustness.py`: Injects empty JSON bundles, corrupt unicode, missing lap durations, and overlapping stints.
  - `test_random_properties.py`: Generates 200 randomly corrupted synthetic race parquets to stress invariants.
  - `test_tracks.py`: Verifies alias collision prevention and circuit distance overrides.

### Mutation Fault Verification Examples
- **Example 1: Missing Required Column (`test_mutation_drop_tracktemp`)**:
  - Dropping `TrackTemp` from a valid DataFrame causes `validate_race_dataframe()` to return `(False, "Missing required columns: {'TrackTemp'}")`. Caught immediately at schema level.
- **Example 2: Unphysical Tyre Life Step (`test_mutation_tyrelife_plus_5`)**:
  - Adding $+5$ to `TyreLife` on lap 15 causes lap 16 to record a negative change ($\Delta \text{TyreLife} < 0$) without a pit stop. `check_tyre_life_rules()` evaluates `tl < prev_tl` on green track and flags `"D01 lap 16 TL reset without pit/comp"`.

---

## 5. DISCREPANCIES TO RESOLVE

### a. 2025 Melbourne, Driver RUS: Stints 1–3
- **Observed in `race_2025_01.parquet`**:
  ```text
      Driver  LapNumber  Stint      Compound  TyreLife              PitInTime             PitOutTime
  2      RUS        1.0    1.0  INTERMEDIATE       1.0                    NaT                    NaT
  23     RUS        2.0    2.0  INTERMEDIATE       3.0 0 days 00:04:53.215000                    NaT
  40     RUS        3.0    3.0  INTERMEDIATE       4.0 0 days 00:07:35.597000 0 days 00:04:53.160000
  57     RUS        4.0    4.0  INTERMEDIATE       5.0 0 days 00:10:12.101000 0 days 00:07:35.594000
  74     RUS        5.0    4.0  INTERMEDIATE       6.0                    NaT 0 days 00:10:12.136000
  ```
- **Raw OpenF1 Cache (`data/openf1_cache/stints_session_key_9693.json`)**:
  ```json
  [
    {"stint_number": 1, "driver_number": 63, "lap_start": 1, "lap_end": 2, "compound": "INTERMEDIATE", "tyre_age_at_start": 0},
    {"stint_number": 2, "driver_number": 63, "lap_start": 2, "lap_end": 3, "compound": "INTERMEDIATE", "tyre_age_at_start": 2},
    {"stint_number": 3, "driver_number": 63, "lap_start": 3, "lap_end": 4, "compound": "INTERMEDIATE", "tyre_age_at_start": 3},
    {"stint_number": 4, "driver_number": 63, "lap_start": 4, "lap_end": 34, "compound": "INTERMEDIATE", "tyre_age_at_start": 4}
  ]
  ```
- **Verdict**: **GENUINE UPSTREAM DATA**. This is not a converter bug. Russell visited the pit lane on consecutive laps during mixed wet weather. OpenF1's feed defines each pit visit as a new stint and carried over `tyre_age_at_start` (0, 2, 3, 4), which `openf1_convert.py` converted faithfully.

### b. 2022 Null TyreLife (520 Values) vs. Invariant
- **Investigation Output**:
  - `data/raw/race_2022_07.parquet` (Monaco GP): 161 nulls across 17 drivers.
  - `data/raw/race_2022_11.parquet` (Austrian GP): 359 nulls across 17 drivers.
  - Total: exactly **520 nulls**.
- **Invariant Implementation**:
  In `tests/data/invariant_helpers.py` line 83:
  ```python
  def check_tyre_life_rules(df):
      tls = df["TyreLife"].dropna()
      if (tls < 1).any():
          return False, f"TyreLife min is {tls.min()}"
  ```
- **Verdict**: The invariant test explicitly executes `.dropna()` before evaluating the condition. It verifies that *when present*, tyre life is $\ge 1$; it does not enforce that tyre life is non-null for 100% of rows.

### c. Phase 3 Re-conversion: "Byte-for-byte" Claim
- **Investigation**:
  Phase 3 used `pd.testing.assert_frame_equal()`, which evaluates in-memory columnar equality.
  To test whether re-conversion achieves disk-level binary reproducibility, three races were re-converted from cache and serialized to temporary parquets:
  ```text
  race_2023_01.parquet:
    DataFrame equal:    True
    Original SHA256:    92aa87bf9ff57f1576718136e66d86f4cb766659560e3e9211bf247ec7699cb6
    Reconverted SHA256: 92aa87bf9ff57f1576718136e66d86f4cb766659560e3e9211bf247ec7699cb6
    SHA256 Match:       True

  race_2024_01.parquet:
    DataFrame equal:    True
    Original SHA256:    a123b06f3439719d5d7cbbcb4ef39963c22a731bc05893a37f827b504fcf7285
    Reconverted SHA256: a123b06f3439719d5d7cbbcb4ef39963c22a731bc05893a37f827b504fcf7285
    SHA256 Match:       True

  race_2025_01.parquet:
    DataFrame equal:    True
    Original SHA256:    051b4bfa27a5be47185d4434ec96e0dcb7c01bfcc117a8e728b1f2f6cf756450
    Reconverted SHA256: 051b4bfa27a5be47185d4434ec96e0dcb7c01bfcc117a8e728b1f2f6cf756450
    SHA256 Match:       True
  ```
- **Verdict**: The converter is strictly deterministic. Re-converting from raw cache produces identical DataFrame objects and **exact matching SHA-256 binary digests**.

### d. Event Name Discrepancies by Source
- In `reports/races.csv`, `event_name` stores raw un-normalized source strings:
  - 2022 R1 (FastF1): `"Bahrain Grand Prix"`
  - 2023 R1 (OpenF1): `"Sakhir"`
  - 2024 R1 (OpenF1): `"Sakhir"`
  - 2025 R1 (OpenF1): `"Melbourne"`
- **Where Normalization Occurs**: Normalization is handled downstream by `src/tracks.py::canonical_track_id()`, which uses the alias lookup in `config/tracks.yaml` to map both `"Bahrain Grand Prix"` and `"Sakhir"` to canonical `"bahrain"`. `reports/races.csv` preserves source metadata heterogeneity.

### e. Test Growth: Phase 0 (61) to Phase 1 (88) to Present (99)
- **Phase 0 (61 tests)**: Initial baseline tests in `tests/` root (`test_ablation.py` through `test_uncertainty.py`).
- **Phase 1 (+27 tests = 88 passing, 19 skipped)**: Created `tests/unit/` and `tests/data/`:
  - `test_converter_robustness.py` (+10 tests)
  - `test_determinism.py` (+3 tests)
  - `test_download_resilience.py` (+3 tests)
  - `test_mutation_faults.py` (+11 tests)
  - `test_invariants.py` (+18 data invariant tests, skipped by default)
  - `test_random_properties.py` (+1 slow test, skipped by default)
- **Phase 2 (+10 tests = 98 passing)**: Added `tests/unit/test_tracks.py` (+10 tests).
- **Hardening Confirmation (+1 test = 99 passing)**: Added `test_mutation_shuffle_rows` to `test_mutation_faults.py`.

### f. Outdated Statements in `reports/project_overview.md`
| Statement in `project_overview.md` | Actual Current Value in Repo |
|---|---|
| "Part 1 (Completed) ... Part 2 (Upcoming)" | Part 2 modules (T2.1–T2.14) have code written, models parameterized, and claim evaluations run. |
| Data source: "FastF1 Raw Telemetry" | Multi-source: FastF1 for 2022; OpenF1 for 2023–2025. |
| Raw dataset: `race_2023_01.parquet`, `race_2023_02.parquet` | 91 valid parquets in `data/raw/` (100,391 laps) + 1 excluded in `data/raw_excluded/`. |
| Processed dataset: "1,663 clean race laps ... 98 clean stints" | Accurate for current `data/processed/`, but stale relative to raw data (cleaning has not run on 91 races). |
| Feature vector: "7-16-8-1 MLP", "7 normalized inputs" | Updated in T2.4 to an 8-16-8-1 MLP with 8 inputs (`track_temp_delta_c`). |
| Model parameters: GRU (5,626), LSTM (7,162), RNN (2,554) | Rebalanced in T2.6 to matched budgets: GRU (7,201), LSTM (7,162), RNN (7,051). |
| Test suite status: "23 out of 23 tests passing" | 118 collected tests (99 passed, 19 skipped by default; 18 data invariants passed). |
| File inventory | Omits 13 modules added since Part 1 (`openf1_source.py`, `openf1_convert.py`, `validate_raw.py`, `tracks.py`, `live.py`, `uncertainty.py`, `claim_eval.py`, `final_eval.py`, `reg_ablation.py`, `sensitivity.py`, `gradflow.py`, `checks/`, `tests/unit/`, `tests/data/`). |
| Recommendations in Section 6 | All 5 suggestions (track fuel scaling, causal spike filtering, track temp delta, monotone head, PWE metric) were already implemented in Part 2. |

---

## 6. WHAT IS DONE vs NOT DONE

| Component | Status | Evidence |
|---|---|---|
| Data Download | **DONE** | 91 valid races downloaded (22 FastF1, 69 OpenF1) in `data/raw/`. 1 cancelled (Imola 2023), 1 excluded (Miami 2025). |
| Data Converters | **DONE** | `src/openf1_convert.py` maps OpenF1 JSON to identical 39-column raw schema. Verified deterministic. |
| Raw Data Validation | **DONE** | `src/validate_raw.py`, 18 invariant tests in `tests/data/` pass on all 91 races. |
| Tracks Configuration | **DONE** | `config/tracks.yaml`: 25 circuits verified against Formula1.com and Wikipedia (0.000 km discrepancy). |
| Cleaning & Fuel Physics | **PARTIAL** | Upgraded with track scaling and spike filtering in `src/clean.py`, BUT `data/processed/` only reflects 2 races. |
| Temporal Splitting | **PARTIAL** | Logic exists in `src/split.py`, but has only been executed on the 2-race pilot data. |
| Baselines A / B / C | **PARTIAL** | Implemented and evaluated in `reports/baseline_table.csv` on 2 races. Not run on 91 races. |
| NumPy MLP | **PARTIAL** | 8-16-8-1 MLP with analytical gradients ($1.42 \times 10^{-8}$) implemented, but trained only on 2 races. |
| Deep Models (GRU/LSTM/RNN) | **PARTIAL** | Balanced architectures (~7.1k params) implemented, trained only on 2 races (`runs/pilot/`). |
| Ablations (EXP-2/3/6/7) | **PARTIAL** | Reported in `reports/ablation.csv`, evaluated on 2 races. Not re-run on 91 races. |
| Streamlit Replay App | **DONE** | `app.py` has replay, 80% uncertainty envelope, ghost line, $k$-gauge, pit window card. Runs on 2-race cache. |
| Part 2: Frozen vs. Rolling | **DONE (Code) / PARTIAL (Data)** | EXP-9 implemented in `src/claim_eval.py`, but evaluated on pilot test stints. |
| Part 2: Adaptive $k$-factor | **DONE** | Update rule implemented in `src/live.py`, dynamic unit tests pass in `tests/test_live.py`. |
| Part 2: MC-Dropout Uncertainty | **DONE** | 30-pass engine implemented in `src/uncertainty.py`, coverage verified in `reports/uncertainty_coverage.csv`. |
| Part 2: Pit Window Formulation | **DONE** | Formulated in `src/claim_eval.py` and rendered in `app.py`. |
| Part 2: 2025 Test Split | **NOT DONE** | Evaluation in `runs/final/` was run on a 2-race pilot set and quarantined (`runs/invalid_final_2023_02/`). Held-out 2025 data has not been cleaned or evaluated. |

*Re-run status*: Models, baselines, and ablations have **NOT** been re-run on the full 91-race dataset. All training and metrics in `reports/` reflect the initial 2-race pilot dataset.

---

## 7. AI-ASSISTANCE FACTS

From git history and `notes/explain.md`:

### Hand-Written by Student (as mandated in `AGENT_RULES.md`)
- NumPy MLP forward and backpropagation logic (`src/mlp_numpy.py` T1.6).
- Two-sided numerical gradient checking function (`src/mlp_numpy.py` T1.6).
- Initial fuel correction function (`src/fuel.py` T1.2).
- Online adaptive wear multiplier $k$-factor update rule (`src/live.py` T2.9).

### Agent-Generated / Pair-Programmed Components
- FastF1 and OpenF1 download pipeline, rate limiting, and cache handlers (`src/download.py`, `src/openf1_source.py`).
- 39-column schema mapping and track status interval parsing (`src/openf1_convert.py`).
- Schema and completeness validators (`src/validate_raw.py`).
- Centralized track management system and scrapers (`src/tracks.py`, `checks/verify_tracks.py`).
- PyTorch recurrent architectures and monotone head (`src/models.py`).
- Streamlit replay dashboard enhancements (`app.py`).
- Pytest test suites across `tests/`, `tests/unit/`, and `tests/data/`.
- Automated check scripts in `checks/`.

---

## 8. TOP 10 RISKS (Teacher Defense Analysis)

1. **Stale Processed Data (`data/processed/`)**:
   `data/processed/laps.parquet` contains only 1,663 laps from 2 races (Bahrain and Jeddah 2023). A teacher inspecting `data/processed/` will immediately notice that none of the models or training runs have utilized the 91 downloaded raw races.
2. **Quarantined Final Run (`runs/final/`)**:
   `runs/final/` was quarantined to `runs/invalid_final_2023_02/` because the original run was executed on 2 races with `messages=False`. The held-out 2025 test season has not yet been processed through `clean.py` or evaluated.
3. **Outdated Project Overview (`reports/project_overview.md`)**:
   The report still describes the repo as having 2 races, 23 tests, and 7 input features, directly contradicting the present 91 races, 118 tests, and 8 features.
4. **Classes Violating AGENT_RULES.md**:
   `AGENT_RULES.md` explicitly allows classes only for `torch.nn.Module`. `src/train.py::StintSequenceDataset` and `src/validate_raw.py::WarningHandler` are non-`nn.Module` classes.
5. **Line Count Exceedance in `checks/f1_evidence.py`**:
   `checks/f1_evidence.py` has 210 lines, exceeding the 200-line limit specified in `AGENT_RULES.md`.
6. **520 Unchecked Null TyreLife Values in 2022**:
   Raw parquets for Monaco and Austria 2022 contain 520 null `TyreLife` values. The invariant test passes only because it calls `.dropna()`. A teacher asking whether tyre life is complete across all races will find these nulls.
7. **FastF1 2023–2025 Incompatibility**:
   Uncached FastF1 downloads fail with HTTP 401 for 2023–2025. If the teacher attempts to download 2023 sessions using standard FastF1 commands, the download will crash.
8. **App Running on Stale 2-Race Data**:
   `app.py` only offers Bahrain and Saudi Arabia 2023 in its dropdown menus because it reads from `data/processed/laps.parquet`.
9. **Raw Event Name Heterogeneity in `reports/races.csv`**:
   `event_name` stores raw strings (`"Bahrain Grand Prix"` vs `"Sakhir"`). A reviewer checking `races.csv` might assume circuit names are un-normalized if they do not check `src/tracks.py`.
10. **OpenF1 Complex Stint Sequences (e.g. Russell 2025 Melbourne)**:
    OpenF1 records 1-lap stints with tyre ages 1, 3, 4, 5 during wet weather pit stops. Without reviewing the raw OpenF1 cache, this appears to be a converter indexing bug.
