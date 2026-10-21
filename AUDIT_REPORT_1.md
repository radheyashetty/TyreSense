# TyreSense Independent Audit Report
**Date:** 2026-10-18  
**Auditor Role:** Independent Academic & Technical Reviewer  
**Audit Scope:** End-to-end audit (data ingestion, preprocessing, features, models, experiments, evaluation, web app UI, and report)  

---

## 1. Audit Check Table

| ID | Check Description | Status | Evidence (File / Line / Command Output) | Severity |
| :--- | :--- | :---: | :--- | :---: |
| **A1** | `requirements.txt` pins exact versions; fresh venv install works | **FAIL** | `requirements.txt:11` specifies `torch==2.14.1+cpu`. Running `pip install --dry-run "torch==2.14.1+cpu"` fails with `ERROR: Could not find a version that satisfies the requirement torch==2.14.1+cpu` (standard PyPI lacks `+cpu` local tag without PyTorch index). Environment is currently running Python 3.14.3 with installed `torch==2.13.0+cpu`. | **Major** |
| **A2** | README commands for data, train, run app, test on existing cache with no network | **FAIL** | `README.md` lines 51–70 only document `pip install -r requirements.txt`, `pytest tests/`, and `streamlit run app.py`. There are no documented CLI commands or Makefile targets to ingest data or train models. | **Minor** |
| **A3** | Config holds seeds; two identical runs give identical metrics to 4 decimals | **PASS** | `config.yaml:42` specifies `seed: 42`. Running 1 epoch of GRU training twice on CPU with seed 42 yielded identical validation metrics: `Run 1: {'mae_h1': 0.9435, 'mae_h5': 1.1538, 'mae_h10': 1.1047}`, `Run 2: {'mae_h1': 0.9435, 'mae_h5': 1.1538, 'mae_h10': 1.1047}`. | None |
| **A4** | `.gitignore` covers data, cache, checkpoints; `git ls-files` shows no F1 data or secrets | **FAIL** | `.gitignore` covers `data/`, `fastf1_cache/`, `.fastf1_cache/`, but does NOT cover `checkpoints/` or `*.pt`/`*.pth`. `git ls-files` confirms no raw or cached F1 data or keys are tracked. | **Minor** |
| **B1** | `reports/races.csv`: races per season, failed/missing, 2025 completion | **FAIL** | `reports/races.csv`: 2022 (22 found, 0 downloaded), 2023 (22 found, 2 downloaded), 2024 (24 found, 0 downloaded), 2025 (24 found, 0 downloaded). The 2025 season has 0 downloaded races and is completely incomplete. | **Major** |
| **B2** | Parquet data checks: counts, nulls, lap times, duplicate rows | **PASS** | `laps.parquet`: 1,663 rows (all 2023 season; Bahrain: 877, Saudi Arabia: 786; Hard: 937, Soft: 407, Medium: 319). 0 nulls across all 14 columns. `stints.parquet`: 98 stints (0 nulls). `lap_time_fc_s`: min 91.68 s, max 100.73 s, mean 96.03 s. Laps < 60 s or > 200 s: 0. Duplicate `(race_id, driver, lap_number)`: 0. | None |
| **B3** | `reports/attrition.csv`: arithmetic reconciliation and match to `laps.parquet` | **FAIL** | `reports/attrition.csv` aggregates all races together without a season breakdown. Filters 1–7 reconcile exactly (1999 - 40 - 150 - 66 - 0 - 0 - 80 = 1663). However, row 8 (`flag_spikes`) lists `rows_dropped: 3, rows_remaining: 1663` ($1663 - 3 \ne 1663$). Spikes are flagged into `gap_mask`, not dropped, making the column header `rows_dropped` misleading. | **Minor** |
| **B4** | Fuel formula recomputation on 20 random laps; circuit length verification | **PASS** | `config.yaml` lines 61–84 contain lengths for all circuits in data (Bahrain: 5.412 km, Saudi Arabia: 6.174 km). Hand recomputation of $t_{\text{fc}} = t - (0.03 \times \text{len} / 5.0) \times (N_{\text{tot}} - N_{\text{lap}})$ across 20 random laps produced a maximum difference of `0.000000 s`. | None |
| **B5** | Spike filter: percentage flagged per season, 5 examples with neighbours, check non-SC | **PASS** | `flag_spikes_offline` flagged exactly 3 laps out of 1,663 (`0.180%`, well below 10% threshold). Rows 604 (Hulkenberg L46, +2.6s), 753 (Sainz L45, +2.2s), 1272 (Tsunoda L46, +2.4s) all had `track_status == '1'` (green flag), clean recovery on the next lap, and were not in/out laps or SC/VSC. | None |
| **B6** | Absence of wet races, lap 1, in-laps, out-laps, SC/VSC, and stints < 8 clean laps | **PASS** | Queries on `laps.parquet`: wet compounds = 0, `lap_number == 1` = 0, `stint_lap == 0` = 0, `track_status` containing 4/5/6/7 (SC/VSC) = 0, stints with length < 8 clean laps = 0 (minimum stint length is 8). | None |
| **B7** | `tyre_age` equals true FastF1 `TyreLife`; `stint_lap` starts at 1 | **PASS** | Merging `laps.parquet` against raw FastF1 race parquets revealed 0 mismatches between `tyre_age` and `TyreLife`. All 98 stints have `min(stint_lap) == 1`. | None |
| **B8** | Recompute usable life for 10 stints; report overall and per-compound censoring | **PASS** | Recomputed usable life and censoring across 10 random stints matched `stints.parquet` with 0 mismatches. Overall censored share: 27.55% (27/98). Hard: 23.91% (11/46), Medium: 52.63% (10/19), Soft: 18.18% (6/33). | None |
| **C1** | Disjoint race_ids across train, val, and test splits | **FAIL** | `config.yaml` sets train: [2022, 2023], val: [2024], test: [2025]. Running `split_by_race()` on `laps.parquet` puts all 1,663 laps into train, and **0 laps into val and test**. In `train.py`, validation was hardcoded to `2023_02`, which was later also used as the test benchmark in `final_eval.py`. | **Blocker** |
| **C2** | Scalers, baseline curves, and track vocabulary fitted on training races only | **FAIL** | `src/split.py` lines 25–56 implement train-only fitting with unseen tracks mapped to `unk: 0`. However, in `src/train.py` line 58, `build_sequences_from_laps(laps_df)` calls `fit_baseline_curves(laps_df)` directly on whatever split is passed, re-fitting curves on the validation/test data rather than using train curves. | **Major** |
| **C3** | Final stint length and `life_laps` are never model inputs | **PASS** | `src/features.py:79-91` constructs `track`, `compound`, `tyre_age`, `lap_fraction`, `relative_pace`, `track_temp`, `track_status`, `gap_mask`. Neither final stint length nor `life_laps` is ever included. | None |
| **C4** | Run `test_leakage.py` and run extra lap t+1 perturbation check on 5 stints | **PASS** | `pytest tests/test_leakage.py` passed (5/5). The independent check in `reports/audit/check_leakage_extra.py` perturbed lap $t+1$ by $+5.0\text{ s}$ across 5 stints; max degradation change at lap $t$ was $0.00000000\text{ s}$ and spike flags remained identical. | None |
| **C5** | App and replay code only receive laps up to the current lap | **PASS** | `app.py:160` strictly slices `observed = full_stint[full_stint["stint_lap"] <= curr_lap]`, and `compute_live_k` strictly loops over `range(min(curr_lap, len(full_stint)))`. | None |
| **D1** | Run `pytest` fully; report passed, failed, skipped, and time | **PASS** | Executed `python -m pytest -v`: **46 passed, 0 failed, 0 skipped** in 11.49 seconds. | None |
| **D2** | Parameter counts within 5% across RNN, LSTM, GRU; input size matches feature list | **FAIL** | Parameter counts are matched within 2.58% (GRU: 5,626; LSTM: 5,582; RNN: 5,726). However, model input dimension is 14 (7 numeric + 4 track + 3 compound). The 8th feature (`track_temp_delta_c`) added in T2.4 was **never wired into `train.py`**, which still uses 6 numeric features + current residual = 7. | **Major** |
| **D3** | NumPy MLP gradient check (< 1e-6) and PyTorch loss gap (< 5%) | **PASS** | Analytical vs numerical gradient check: 7-input max relative error = $1.42 \times 10^{-8}$; 8-input max relative error = $3.40 \times 10^{-7}$ (both $< 10^{-6}$). Relative loss gap to PyTorch after 10 SGD steps = $5.13 \times 10^{-8}$ (0.0000%, well within 5%). | None |
| **D4** | `ablation.csv` has EXP-1 to EXP-10, prior hypotheses, seeds (3 or 5), std dev, runs/ | **FAIL** | 1. `ablation.csv` omits EXP-1 (in test), EXP-8 (`gradflow.csv`), and EXP-9 (`exp9_core_claim.csv`). 2. Hypotheses were committed in the same git commit as the results (`39e2aee`, `fbfd6f5`). 3. EXP-7 reports only means (no standard deviations). EXP-4/5 are single runs without std dev. 4. `runs/` only contains `pilot` and `final`; configs for EXP-2 through EXP-10 were not saved to `runs/`. | **Major** |
| **D5** | Check results against hypotheses; identify negative/surprising results; report honesty | **FAIL** | In EXP-9, ROLLING and Model E are **worse than FROZEN at Lap 5** (PWE 3.94 vs 3.70) and **substantially worse at Lap 7** (PWE 4.53 vs 3.70), and Model E gave **0.00 lap improvement over ROLLING**. `final_report.md:85` cherry-picked only Lap 3 (-0.71 laps) and completely concealed the negative results at Laps 5 and 7. In EXP-7, vanilla RNN beat GRU at $h=10$ (1.124 s vs 1.332 s). | **Blocker** |
| **D6** | Baselines (A, B, C, GRU) evaluated on same stints/metrics; no mixed splits | **FAIL** | `reports/baseline_table.csv` contains only Models A, B, and C (786 laps of `2023_02`). Model D (GRU) was omitted from `baseline_table.csv`. Furthermore, the MAE numbers in `baseline_table.csv` (Model A $h=1$: 1.656 s) contradict those reported in `ablation.csv` (Model A $h=1$: 0.759 s) and cited in `final_report.md`. | **Major** |
| **D7** | Training vs validation MAE gap for final GRU (overfitting assessment) | **FAIL** | Final GRU evaluation: Train MAE ($h=1$: 0.4116 s, $h=5$: 0.4203 s, $h=10$: 0.5097 s) vs Val MAE ($h=1$: 1.0909 s, $h=5$: 1.2087 s, $h=10$: 1.3216 s). Generalization gap is **+0.679 s at $h=1$ and +0.812 s at $h=10$** (validation error is 2.65x training error), indicating severe overfitting to the single training race. | **Major** |
| **E1** | `src/live.py`: normalized update, smoothed over 3, clipped [0.5, 2.0], $\ge$ lap 3, no spike update | **PASS** | `src/live.py:6-16, 43`: update is normalized ($g \cdot \text{err} / (g^2 + 0.05)$), smoothed over last 3 values (`k_history[-3:]`), clipped to `[0.5, 2.0]`, guarded by `stint_lap >= 3`, and skipped if `is_spike == True`. | None |
| **E2** | `test_live.py`: report lap where $k$ comes within 10% of true rate | **PASS** | Running the synthetic noisy wear benchmark ($\text{true\_k} = 1.5$, $\sigma = 0.1$) converges within 10% ($[1.35, 1.65]$) at **Stint Lap 9** ($k = 1.363$, error = 9.12%). | None |
| **E3** | FROZEN fixed at lap 2; ROLLING recomputed every lap; identical models/inputs | **PASS** | `src/claim_eval.py:43, 47`: FROZEN calls `forecast_stint_pace(..., lap_idx=1)` (stint lap 2); ROLLING calls `forecast_stint_pace(..., lap_idx=target_lap - 1)`. Both use identical model functions and inputs. | None |
| **E4** | Uncertainty coverage at 50, 60, 70, 80, 90%; fit scale factor on train folds only | **FAIL** | `src/uncertainty.py:72-82` generated synthetic random Gaussian inputs (`torch.randn(1, 10, 7)`) instead of real validation stints. On random inputs, degradation never hit $+1.0\text{ s}$, setting $p10=p50=p90=21$. `uncertainty_coverage.csv` records `raw_width: 0.0, scaled_width: 0.0`. Coverage at 50, 60, 70, 90% was never evaluated. `final_report.md:91` fabricated "$\pm 1.8\text{ laps}$". | **Blocker** |
| **E5** | Pit window $[t + P_{10} - 2, t + P_{50}]$; shown only when $P_{10}, P_{90}$ exist, never before lap 3 | **FAIL** | In `app.py:168`, pit window is hardcoded as `[Lap curr_lap + max(1, rem_int - 5), Lap curr_lap + rem_int]`, using a fixed heuristic rather than MC-Dropout percentiles. Furthermore, it is displayed on Stint Laps 1 and 2 (not hidden before lap 3). | **Major** |
| **E6** | PWE computed on uncensored stints only; censored scored separately; 10 stints check | **FAIL** | PWE is computed only on uncensored stints, but censored stints were **never scored separately**. Crucially, when forward forecasts do not cross $+1.0\text{ s}$, `compute_pwe` returns `NaN`. At Lap 3, only **7 out of 71 stints** had valid pairs for the paired bootstrap; at Lap 5, only 6 stints; at Lap 7, only 5 stints. The bootstrap CI was calculated on $N \le 7$ stints without disclosing the massive sample attrition. | **Blocker** |
| **F1** | `runs/final/` exists, contains one run, records commit hash and date | **FAIL** | `runs/final/metrics.json` exists and records commit hash `b52b92063a888ae5479b20b1dc87faa804548baf`, but **does not record the date/timestamp**. | **Minor** |
| **F2** | `git log` shows final evaluation ran once with no later code changes to src/models/config | **FAIL** | 1. The "test run" in `src/final_eval.py` evaluated `2023_02` (which was already used as the validation set in `train.py` and `pilot`), NOT the 2025 test season. 2. `src/final_eval.py` was committed and re-run in commit `fb88815` after the initial final evaluation commit `d785e7e`. | **Blocker** |
| **F3** | Test result reported in report with no cherry-picking | **PASS** | Metrics in `runs/final/metrics.json` ($h=1: 1.091\text{ s}, h=5: 1.209\text{ s}, h=10: 1.322\text{ s}$) match `reports/final_report.md` Section 4 exactly. | None |
| **G1** | Start app with documented command; save 4 screenshots in `reports/audit/screenshots/` | **PASS** | Captured and saved: `first_load.png`, `stint_lap3.png`, `stint_mid.png`, `stint_last.png` in `reports/audit/screenshots/`. | None |
| **G2** | Check controls: season, race, driver, stint, model; back, play, forward; slider | **FAIL** | 1. There is **no Season selector** in `app.py`. 2. There is **no Play button** (only Back and Forward). 3. Selecting "Model D (GRU)" runs the exact same code as "Model A (Baseline)" (`k=1.0`); the PyTorch GRU model is never executed in the app. | **Major** |
| **G3** | Compare screen numbers for 3 stints with code computation | **PASS** | Screen metrics (Assumed fuel, $k$, Remaining Life, Pit Window) matched direct code computation across `2023_01_VER_3`, `2023_01_ALO_12`, and `2023_02_GAS_5`. | None |
| **G4** | Band hidden before lap 3; ghost line visible; +1.0s line drawn; fuel card says "assumed" | **FAIL** | The uncertainty band is **not hidden before lap 3** (drawn at lap 1 and 2). Fuel card correctly says "Assumed fuel on board", ghost line is drawn, and +1.0 s cliff line is drawn. | **Minor** |
| **G5** | Missing or bad data shows clear message, not a crash | **FAIL** | If a filter combination results in no stints, `app.py:170` attempts `observed["total_laps"].iloc[-1]`, throwing an unhandled `IndexError` rather than rendering a user-friendly error message. | **Minor** |
| **G6** | Latency benchmarks: first load < 3 s, each lap step < 1 s | **PASS** | Measured: first data load = `0.064 s` (< 3.0 s target); single lap step compute + render = `0.089 s` (< 1.0 s target). | None |
| **G7** | Narrow window width test | **FAIL** | At 500px width (`reports/audit/screenshots/narrow_width.png`), Streamlit collapses the sidebar into a closed menu and squeezes 6 metric columns into unreadable truncated slivers. | **Minor** |
| **H1** | All files $\le 200$ lines, functions $\le 30$ lines, no lambdas, no unlisted libraries | **PASS** | AST check across all 19 files in `src/` and `app.py`: 0 line count violations, 0 function length violations, 0 lambdas. All imports strictly adhere to allowed libraries. | None |
| **H2** | Every number in report and README traces to files in `reports/` or `runs/` | **FAIL** | 1. `README.md:15` cites stale parameter counts: LSTM (7,162) and RNN (2,554) instead of matched 5,582 and 5,726. 2. `README.md:42, 75` claims 23 tests instead of 46. 3. `final_report.md:91` claims interval width of "$\pm 1.8\text{ laps}$", which appears nowhere in `uncertainty_coverage.csv` (which has 0.0). 4. Numbers in `reports/baseline_table.csv` contradict `reports/ablation.csv`. | **Major** |
| **H3** | Wording check: "Bayesian", "Kalman", "temperature scaling", "guarantee", "proves" | **FAIL** | 1. `reports/final_report.md:87` claims Lap 3 telemetry "Proves" an improvement despite the 95% bootstrap CI $[-2.71, 1.43]$ spanning zero. 2. `reports/project_overview.md` uses "Bayesian" (line 14), "temperature scaling" (line 122), and "guarantee" (line 122, 234). 3. Usable life is not explicitly stated as a chosen definition in `final_report.md`. | **Major** |
| **H4** | Academic credit for cleaning, fuel value, and external papers | **FAIL** | `reports/final_report.md` has no bibliography or references section. Neither the cleaning methodology, the fuel correction source ($0.03\text{ s/lap}$), nor any external papers are cited. | **Major** |
| **H5** | AI assistance disclosed; `notes/explain.md` has entry for every task | **FAIL** | `notes/explain.md` has complete entries for all 23 tasks (T1.1–T1.9, T2.1–T2.14). However, AI assistance is **not disclosed** anywhere in `reports/final_report.md`. | **Minor** |
| **H6** | Syllabus map in report; explicit statement that Unit 4 (CNNs) is not covered | **FAIL** | `reports/final_report.md` does NOT include a syllabus map, does not map experiments to units, and does not state that Unit 4 (CNNs) is omitted. | **Major** |

---

## 2. Blockers (Invalidating Results)

1. **Test Set Reuse & Unopened 2025 Season (Check C1, F2)**:
   The PRD explicitly mandated training on 2022–2023, validating on 2024, and holding out 2025 locked until the final evaluation. In reality, **only two races from 2023 exist in the entire dataset** (`2023_01` and `2023_02`). Race `2023_02` (Saudi Arabia) was used as the validation set during development (`train.py`, `pilot`, `baseline_table.csv`) and was **re-used as the "held-out final test set" in `src/final_eval.py`**. The true 2025 test split was never ingested or evaluated.
2. **Selective Reporting & Concealment of Negative Results in Core Claim (Check D5)**:
   In EXP-9 (`reports/exp9_core_claim.csv`), ROLLING and Model E are **inferior to the FROZEN baseline** at Stint Lap 5 (PWE 3.94 vs 3.70) and Stint Lap 7 (PWE 4.53 vs 3.70). Furthermore, online adaptive Model E provided **$0.00\text{ laps}$ improvement over un-adapted ROLLING** across all laps. `reports/final_report.md` selectively reported only Lap 3 (claiming it "proves" superiority) and completely omitted Laps 5 and 7 and the zero-delta of Model E.
3. **Statistical Invalidity of PWE Bootstrap Due to NaN Attrition (Check E6)**:
   In `src/claim_eval.py`, when a 20-lap forecast does not reach the $+1.0\text{ s}$ degradation cliff, `compute_cliff_life` returns `None`, resulting in `NaN` PWE. Out of 71 uncensored stints, only **7 stints at Lap 3, 6 stints at Lap 5, and 5 stints at Lap 7** had valid predictions for both FROZEN and ROLLING. Running a 1,000-resample paired bootstrap on a sample size of $N=7$ (discarding 90% of stints) without disclosing this attrition makes the reported confidence intervals statistically uninterpretable.
4. **Synthetic Fabrication in Uncertainty Quantification (Check E4, H2)**:
   In `src/uncertainty.py`, `sample_stint_intervals()` evaluated the GRU model on random synthetic tensors (`torch.randn(1, 10, 7)`) rather than real race stints. Because random inputs never trigger degradation, $P_{10}=P_{50}=P_{90}=21$, resulting in `raw_width = 0.0` and `scaled_width = 0.0` in `reports/uncertainty_coverage.csv`. `reports/final_report.md` reported that scaling balanced interval width to "$\pm 1.8\text{ laps}$", a number with zero empirical basis in the codebase or report artifacts.

---

## 3. Prioritized Fix List

### Blocker Fixes
1. **Disclose Data Constraints Honestly in the Report**:
   State plainly that due to FastF1 rate-limiting and 2025 calendar availability, the evaluation was conducted across the 2 available 2023 dry Grand Prix rounds (`2023_01` as train, `2023_02` as validation/test). Remove claims that 2025 was evaluated as an independent season.
2. **Report Full EXP-9 Results Without Cherry-Picking**:
   Update `final_report.md` Section 3 (EXP-9) to include the full table from `reports/exp9_core_claim.csv` (Laps 3, 5, and 7). State honestly that:
   - Early in the stint (Lap 3), ROLLING provides an advantage (-0.71 laps, 95% CI [-2.71, 1.43]).
   - Later in the stint (Laps 5 and 7), ROLLING degrades relative to FROZEN (+0.24 laps and +0.83 laps error).
   - The online error-correction multiplier ($k$) provides identical PWE to ROLLING on this dataset because baseline quadratic offsets absorb the primary pace shift.
3. **Resolve PWE Censoring and Sample Size Attrition**:
   Modify `compute_pwe()` to assign the maximum horizon (20 laps) or score censored/non-crossing forecasts under a survival metric rather than discarding them as `NaN`. Explicitly state the effective sample size $N$ in the report.
4. **Re-Run Uncertainty Calibration on Real Validation Stints**:
   Update `src/uncertainty.py` to pass actual validation stint sequences from `2023_02` through `mc_dropout_stint_pass()` instead of random noise tensors. Re-generate `uncertainty_coverage.csv` and report the true empirical widths and coverage percentages across 50, 60, 70, 80, and 90%.

### Major Fixes
5. **Connect Model D (GRU) to Streamlit UI (`app.py`)**:
   Import `create_gru_model` and load the trained weights in `app.py`. When the user selects "Model D (GRU Recurrent)", run the actual neural network rather than falling back to the polynomial curve.
6. **Wire 8th Feature (`track_temp_delta_c`) into `train.py`**:
   Update `build_sequences_from_laps()` in `src/train.py` line 65 to include `track_temp_delta_c` so that the PyTorch models actually utilize the 8 features developed in T2.4.
7. **Fix Baseline Table Discrepancies**:
   Reconcile `reports/baseline_table.csv` and `reports/ablation.csv` so that baseline model numbers match across all reports, and add Model D (GRU) to `baseline_table.csv`.
8. **Fix Inaccurate Wording in Report & Overview**:
   Remove the word "Proves" from `final_report.md:87` (replace with "Suggests" or "Indicates"). In `project_overview.md`, replace "Bayesian" with "normalized error correction" and remove "temperature scaling" and "guarantees".
9. **Add Syllabus Map & Paper Citations**:
   Add a 1-page section to `final_report.md` mapping Units 1, 2, 3, and 5 to their corresponding experiments, explicitly stating that Unit 4 (CNNs) is not covered. Add formal citations for FastF1 and the 0.03 s/lap fuel rule.

### Minor Fixes
10. **Fix `requirements.txt` PyTorch Pin**:
    Change `torch==2.14.1+cpu` to `torch==2.13.0` or standard PyPI release `torch>=2.0.0` with installation instructions for CPU wheels.
11. **Hide Uncertainty Band and Pit Window Before Lap 3 in `app.py`**:
    Add `if curr_lap >= 3:` guards around the uncertainty band in Plotly and the pit window metric card.
12. **Update Stale Numbers in `README.md`**:
    Update test count to 46 (from 23) and parameter counts to GRU 5,626, LSTM 5,582, RNN 5,726.
13. **Add `checkpoints/` to `.gitignore`**.

---

## 4. Questions for the Student

1. **Season Availability**: Did you attempt downloading the full 2024 season from FastF1, or did network/time constraints restrict you to the two cached 2023 races?
2. **Model D in App**: Was the omission of the PyTorch GRU model from `app.py` an intentional choice to preserve the $< 1.0\text{ s}$ interactive latency budget, or an unfinished integration?
3. **Model E $k$-Factor Inactivity**: In EXP-9, Model E produces identical PWE numbers to uncorrected ROLLING. Did you observe whether the error-correction rule triggered on actual stints, or did the quadratic baseline shift absorb the residual pace discrepancy before $k$ could adapt?
4. **Course Requirements**: Does your university/instructor require an AI assistance declaration, and do they accept projects that omit Unit 4 (CNNs) in favor of deep sequence modeling?

---

## 5. Final Verdict

**Is the project ready to submit? NO.**

While TyreSense exhibits outstanding software engineering discipline—100% compliant with strict line-length rules, zero lambdas, cleanly structured modular code, and 46 passing tests—it currently contains **critical scientific discrepancies that would fail a rigorous external viva**. The "held-out test season" evaluated the exact same 2023 race previously used for validation, the core claim in the final report cherry-picked Lap 3 while concealing that the model was worse than the baseline at Laps 5 and 7, uncertainty quantification was fitted on synthetic random Gaussian noise rather than real telemetry, and the Streamlit dashboard does not actually run the deep learning model. Implementing the fixes outlined above (chiefly honest reporting of negative findings, re-running uncertainty on real stints, and connecting the GRU to the UI) will transform TyreSense into an academically bulletproof, highly commendable submission.
