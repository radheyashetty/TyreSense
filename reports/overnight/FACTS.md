# FACTS LEDGER - TYRESENSE OVERNIGHT RUN

Generated: 2026-10-21T04:20:00+05:30
Branch: overnight-run
Base commit: 457ccd9
Target Python Interpreter: C:\Python314\python.exe (Python 3.14.3)
Test Runner Interpreter: C:\Users\radhe\AppData\Local\Programs\Python\Python310\python.exe (Python 3.10.11)

---

## 1. Raw Telemetry Integrity
- Total raw telemetry parquets: 92 (91 in data/raw/, 1 in data/raw_excluded/)
- Comparison against reports/testing/raw_hashes_before.csv:
  - Total files matched: 92 / 92 (100.0%)
  - Mismatches: 0
  - Missing: 0

---

## 2. Dataset Split Sizes
- Total Races Processed: 79 valid clean races (12 dropped as wet races from 91 downloaded)
  - Train Split (2022–2023): 38 races, 33,269 clean laps, 1,756 stints (578 uncensored, 1,178 censored; 67.08% censored)
  - Val Split (2024): 21 races, 20,028 clean laps, 903 stints (362 uncensored, 541 censored; 59.91% censored)
  - Test Split (2025 - Locked): 20 races, 19,508 clean laps, 865 stints (295 uncensored, 570 censored; 65.90% censored)
- R3 Test Lock Assertion: Verified in src/split.py:assert_no_2025_data and tests/test_split.py:test_r3_test_lock_assertion.

---

## 3. Label Diagnostic (Persistent 2-Lap Cliff Rule >= 4 Laps)
- Usable life defined as: first lap >= 4 where degradation >= +1.0s for 2 consecutive laps.
- Uncensored stints in Training (N=578):
  - SOFT (N=129): mean=9.7, med=9.0, p10=5.0, p25=7.0, p75=12.0, p90=14.2
  - MEDIUM (N=226): mean=11.7, med=11.0, p10=5.0, p25=7.0, p75=15.0, p90=20.0
  - HARD (N=223): mean=14.7, med=14.0, p10=6.0, p25=8.0, p75=19.0, p90=24.0
  - Cliffs at lap <= 3: 0 (0.00%)
- Uncensored stints in Validation (N=362):
  - Mean=13.3, median=12.0, min=4.0, max=41.0
  - Stints with true remaining life <= 20 at lap 5: 339 (93.6%)
  - Stints with true remaining life > 20 at lap 5: 23 (6.4%)

---

## 4. Model E Multiplier Investigation
- Evaluated on all 903 validation stints:
  - Stints with k moving > 0.05: 0 (0.00%)
  - Stints with k remaining identically 1.000: 903 (100.0%)
- Mathematical Root Cause:
  In simulate_modes_at_lap, base_preds is computed via predict_causal_baseline.
  For laps i < target_lap, g_t = act_t - base_t.
  At initial k = 1.0, predicted lap time is base_t + 1.0 * g_t = act_t.
  Hence err = act_t - pred = 0.0 identically, producing zero update gradient.

---

## 5. Baselines Table (Matched Capacity Evaluation on 2024 Val)
- Model A (Quadratic Baseline Curve, 3 params):
  - h=1 MAE: 1.767 s
  - h=5 MAE: 1.832 s
  - h=10 MAE: 2.066 s
- Model B (Ridge Regression, 10 params):
  - h=1 MAE: 1.278 s
  - h=5 MAE: 1.661 s
  - h=10 MAE: 1.936 s
- Model C (XGBoost, 500 trees):
  - h=1 MAE: 0.955 s
  - h=5 MAE: 1.495 s
  - h=10 MAE: 1.958 s
- Baseline Mean:
  - h=1 MAE: 1.924 s
  - h=5 MAE: 2.012 s
  - h=10 MAE: 2.291 s
- Model D (GRU, 5,626 params, 3 seeds mean):
  - h=1 MAE: 0.939 s
  - h=5 MAE: 1.012 s
  - h=10 MAE: 1.219 s
- Model D (LSTM, 5,582 params, 3 seeds mean):
  - h=1 MAE: 0.953 s
  - h=5 MAE: 0.996 s
  - h=10 MAE: 1.204 s
- Model D (RNN, 5,726 params, 3 seeds mean):
  - h=1 MAE: 0.947 s
  - h=5 MAE: 1.117 s
  - h=10 MAE: 1.453 s
- Matched capacity parameter delta:
  - (5726 - 5582) / 5582 = 2.58% (<= 5.0% threshold)
- Final GRU Generalization Gap:
  - Train: h=1: 0.473 s, h=5: 0.510 s, h=10: 0.563 s
  - Val: h=1: 0.827 s, h=5: 0.946 s, h=10: 1.165 s
  - Generalization Gap: h=1: +0.354 s, h=5: +0.436 s, h=10: +0.602 s

---

## 6. Core Claim PWE (reports/overnight/exp9_results.csv)
- Evaluated on uncensored 2024 validation stints at stint laps 3, 5, 7, 10:
  - Lap 3 (N=326):
    - FROZEN capped: 9.67 laps
    - ROLLING capped: 9.42 laps
    - Always-20: 11.35 laps
    - Always-Train-Median: 4.17 laps
    - Rolling advantage: +0.25 laps
    - 95% Stint CI: [-0.11, 0.61] laps (spans zero)
    - 95% Cluster CI: [0.03, 0.47] laps
    - Crossing difference share: 30.4%
    - Horizon 1 MAE: FROZ=0.608 s, ROLL=0.414 s
    - Censored survival consistency: FROZ=54.5%, ROLL=56.0%, Always-20=64.9%
  - Lap 5 (N=339):
    - FROZEN capped: 10.72 laps
    - ROLLING capped: 8.83 laps
    - Always-20: 12.81 laps
    - Always-Train-Median: 4.38 laps
    - Rolling advantage: +1.89 laps
    - 95% Stint CI: [1.30, 2.50] laps
    - 95% Cluster CI: [1.26, 2.54] laps
    - Crossing difference share: 43.4%
    - Horizon 1 MAE: FROZ=0.827 s, ROLL=0.385 s
    - Censored survival consistency: FROZ=62.8%, ROLL=65.4%, Always-20=73.8%
  - Lap 7 (N=346):
    - FROZEN capped: 11.82 laps
    - ROLLING capped: 8.22 laps
    - Always-20: 14.31 laps
    - Always-Train-Median: 4.36 laps
    - Rolling advantage: +3.60 laps
    - 95% Stint CI: [2.86, 4.35] laps
    - 95% Cluster CI: [2.65, 4.42] laps
    - Crossing difference share: 49.1%
    - Horizon 1 MAE: FROZ=1.254 s, ROLL=0.532 s
    - Censored survival consistency: FROZ=66.5%, ROLL=69.1%, Always-20=78.6%
  - Lap 10 (N=337):
    - FROZEN capped: 12.77 laps
    - ROLLING capped: 7.61 laps
    - Always-20: 15.90 laps
    - Always-Train-Median: 3.90 laps
    - Rolling advantage: +5.16 laps
    - 95% Stint CI: [4.35, 5.99] laps
    - 95% Cluster CI: [3.74, 6.43] laps
    - Crossing difference share: 58.2%
    - Horizon 1 MAE: FROZ=1.488 s, ROLL=0.512 s
    - Censored survival consistency: FROZ=71.5%, ROLL=72.1%, Always-20=82.9%

---

## 7. Uncertainty Coverage Calibration (reports/uncertainty_coverage.csv)
- Target 50%: Raw Coverage = 8.8% (width 0.74 laps), Scaled = 14.9% (width 2.96 laps, s=4.0)
- Target 60%: Raw Coverage = 9.7% (width 0.95 laps), Scaled = 16.3% (width 3.81 laps, s=4.0)
- Target 70%: Raw Coverage = 10.2% (width 1.12 laps), Scaled = 18.2% (width 4.47 laps, s=4.0)
- Target 80%: Raw Coverage = 11.9% (width 1.46 laps), Scaled = 20.4% (width 5.85 laps, s=4.0)
- Target 90%: Raw Coverage = 13.5% (width 1.75 laps), Scaled = 21.3% (width 7.01 laps, s=4.0)

---

## 8. Codebase & Compliance Facts
- Total Python files: 50
- Files with > 200 lines: 0 (0.0%)
- Functions with > 30 lines: 0 (0.0%)
- Lambdas in src/: 0
- Pytest suite: 126 tests collected, 107 passed, 19 skipped (slow/network), 0 failed
- Data invariant tests: 18 passed, 0 failed
