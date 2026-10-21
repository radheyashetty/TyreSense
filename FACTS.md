# TyreSense Measured Facts Ledger (Run 4 Freeze)

**Generated**: 2026-10-21T21:40:00+05:30  
**Branch**: `overnight-run`  
**Base Tag**: `pre-run4` (Commit `2c1bba5`)  
**Freeze Tag**: `pre-test-open`  

---

## 1. Environment & Library Versions
- **Python (Experiment Runner)**: Python 3.14.3 (`C:\Python314\python.exe`)
- **Python (Test Runner)**: Python 3.10.11 (`C:\Users\radhe\AppData\Local\Programs\Python\Python310\python.exe`)
- **PyTorch**: `torch==2.11.0.dev20260217+cpu`
- **FastF1**: `fastf1==3.4.4`
- **Pandas**: `pandas==2.2.3`
- **NumPy**: `numpy==2.2.3`
- **Scikit-Learn**: `scikit-learn==1.6.1`
- **XGBoost**: `xgboost==3.0.0`
- **Plotly**: `plotly==6.0.0`
- **Streamlit**: `streamlit==1.42.0`
- **Pytest**: `pytest==9.0.3`

---

## 2. Dataset Season & Split Counts
Total Races: 91 classified races (92 raw parquets including 1 excluded wet race).
- **Season 2022**: 22 races (18 valid non-wet, 2 excluded wet: Singapore `2022_17`, Japan `2022_18`, 2 failed/missing) | Source: `FastF1`
- **Season 2023**: 23 races (20 valid non-wet, 2 excluded wet: Monaco `2023_06`, Netherlands `2023_13`, 1 failed/missing) | Source: `OpenF1`
- **Season 2024**: 24 races (21 valid non-wet, 3 excluded wet: Canada `2024_09`, Great Britain `2024_12`, Sao Paulo `2024_21`) | Source: `OpenF1`
- **Season 2025 (Locked Test Set)**: 22 races (20 valid non-wet, 1 excluded wet: Great Britain `2025_12`, 1 excluded red-flag: Australia `2025_01`) | Source: `OpenF1`

### Split Lap Counts (`data/processed/laps.parquet`)
- **Train Split (2022–2023)**: 33,269 clean laps (38 non-wet races)
- **Validation Split (2024)**: 20,028 clean laps (21 non-wet races)
- **Locked Test Split (2025)**: 19,508 clean laps (20 non-wet races, held uninspected under R3)

---

## 3. Data Cleaning Pipeline (Execution Order)
From `src/clean.py`:
1. **Wet-Race Exclusion**: Drop entire races where $> 10\%$ of laps have `Rainfall == True` or compound is `INTERMEDIATE` / `WET`.
2. **Missing Lap Time Removal**: Filter out rows with null `LapTime`.
3. **Compound Filtering**: Keep only dry slicers (`SOFT`, `MEDIUM`, `HARD`); drop `None`, null, or unknown compounds.
4. **IsAccurate Telemetry Filter**: For FastF1 (2022), drop laps where `IsAccurate == False`; for OpenF1 (2023–2025) where `IsAccurate` is NaN, retain laps if non-outlier.
5. **Physical Speed Filter**: Retain only laps with $50.0\text{ s} \le \text{LapTime} \le 150.0\text{ s}$.
6. **Circuit-Scaled Fuel Physics Correction**:
   $$c = 0.03 \times \frac{\text{Circuit Length (km)}}{5.303\text{ km}}$$
   $$\text{lap\_time\_fc} = \text{lap\_time} - c \times (\text{total\_laps} - \text{lap\_number})$$
7. **Causal Spike Flagging**: Flag laps $> 1.5\text{ s}$ slower than recent 3-lap rolling median as safety car / traffic anomalies without looking ahead.

---

## 4. Degradation Cliff Label Definition
From `src/clean.py:compute_stint_labels`:
- **Threshold**: Fuel-corrected degradation $\ge +1.0\text{ s}$ above stint baseline pace (mean of laps 1–3).
- **Persistence**: Requires **2 consecutive laps** crossing the threshold.
- **Minimum Lap Guard**: Cliff cannot fire before stint lap 4 (`stint_lap >= 4`).
- **Censoring**: Stints pitting before 2 consecutive crossing laps are marked `censored = True`.

---

## 5. Model Architecture & Parameter Counts
From `src/models.py`:
- **Inputs**: 8 numeric features + 4 track embedding dims + 3 compound embedding dims = 15 total input dims.
- **GRU**: 5,802 parameters (`nn.GRU(input_size=15, hidden_size=32, num_layers=2)`)
- **LSTM**: 5,770 parameters (`nn.LSTM(input_size=15, hidden_size=27, num_layers=2)`)
- **RNN**: 5,864 parameters (`nn.RNN(input_size=15, hidden_size=58, num_layers=2)`)
- **Capacity Parity**: Max deviation across recurrent models is $1.6\%$.

---

## 6. Fair EXP-9 Standardization (`reports/overnight/exp9_official.csv`)
- **Fair Formula**:
  $$\text{PWE}_{\text{abs}} = |\min(L_{\text{pred}}, 22) - \min(L_{\text{true}}, 22)| \quad (H = 22\text{ laps})$$
- **Dynamic Alive-at-t Cohort** ($L_{\text{true}} \ge t$):
  - **Lap 3** ($N=362$ alive, 0 excl, 21 races): FROZEN $8.14$, ROLLING $7.94$, Model E $8.06$, Always-22 $9.34$, Always-Train-Median $4.60$. ROLLING advantage $+0.20\text{ laps}$ (95% stint CI $[-0.12, 0.54]$, 95% cluster CI $[-0.05, 0.46]$, $\text{Var}_{\text{between}} = 1.2368$). Model E Adv: $-0.12$ (CI $[-0.33, 0.07]$). FROZEN crossing share: 0.257. Censored ($N=541$): FROZEN 0.871, ROLLING 0.885, Always-22 1.000.
  - **Lap 5** ($N=345$ alive, 17 excl, 21 races): FROZEN $7.94$, ROLLING $6.97$, Model E $7.07$, Always-22 $8.91$, Always-Train-Median $4.38$. ROLLING advantage $+0.96\text{ laps}$ (95% stint CI $[0.50, 1.39]$, 95% cluster CI $[0.38, 1.42]$, $\text{Var}_{\text{between}} = 0.8051$). Model E Adv: $-0.10$ (CI $[-0.31, 0.10]$). FROZEN crossing share: 0.249. Censored ($N=541$): FROZEN 0.871, ROLLING 0.902, Always-22 1.000.
  - **Lap 7** ($N=315$ alive, 47 excl, 21 races): FROZEN $7.37$, ROLLING $5.60$, Model E $6.11$, Always-22 $8.21$, Always-Train-Median $4.22$. ROLLING advantage $+1.78\text{ laps}$ (95% stint CI $[1.28, 2.28]$, 95% cluster CI $[1.10, 2.29]$, $\text{Var}_{\text{between}} = 1.1892$). Model E Adv: $-0.52$ (CI $[-0.86, -0.20]$, cluster CI $[-0.95, -0.18]$). FROZEN crossing share: 0.244. Censored ($N=541$): FROZEN 0.871, ROLLING 0.891, Always-22 1.000.
  - **Lap 10** ($N=235$ alive, 110 excl, 19 races): FROZEN $5.93$, ROLLING $4.14$, Model E $4.50$, Always-22 $6.24$, Always-Train-Median $4.16$. ROLLING advantage $+1.78\text{ laps}$ (95% stint CI $[1.23, 2.31]$, 95% cluster CI $[0.95, 2.52]$, $\text{Var}_{\text{between}} = 1.6275$). Model E Adv: $-0.35$ (CI $[-0.63, -0.10]$, cluster CI $[-0.95, 0.01]$). FROZEN crossing share: 0.226. Censored ($N=491$): FROZEN 0.876, ROLLING 0.914, Always-22 1.000.
- **Model E Diagnosis & Resolution**:
  - `predict_causal_baseline` applied offset only starting at lap 3 (`stint_lap >= 3`). The difference `base_preds[2] - base_preds[1]` included the full static stint offset (up to $-17.48\text{ s}$ on Zandvoort with global curve), crashing $k$ into lower clamp $0.5$.
  - Fixed in `compute_online_k` using pure incremental quadratic wear `raw_curve[i] - raw_curve[i-1]`.
  - While $k$ is restored to healthy median $1.000$, plain ROLLING remains superior due to lower estimator variance.
- **Conformal Coverage Range**:
  - Evaluated across 5 seeded race splits (seeds 1, 2, 3, 4, 42); 80% nominal coverage mean is $69.7\%$ (range $62.8\%$ - $76.4\%$, with label "interval under-covers; shown for illustration only").
  - Compound training medians: HARD 14, MEDIUM 11, SOFT 9.

---

## 7. Code Verification & Test Integrity
- **Test Suites**:
  - `pytest`: 114 passed, 19 skipped, 1 warning in 25.14s
  - `pytest -m data`: 18 passed, 115 deselected in 50.78s
  - `pytest -m slow`: 1 passed, 132 deselected in 87.05s
- **Raw Telemetry**: 100% bitwise SHA-256 match across all 92 raw parquets against `reports/testing/raw_hashes_before.csv`.
- **2025 Test Set Lockout**: Zero 2025 metric files exist; `assert_no_2025_in_app` verified in data and UI loops.
- **AST Integrity**: 63 codebase files checked, zero violations ($\le 200$ lines/file, $\le 30$ lines/function, 0 lambdas).

---

## 8. Output Artifact Paths
- Baseline Comparison: `reports/baseline_table.csv`
- Ablation Experiments: `reports/ablation.csv`
- Core Claim EXP-9 (Official Fair): `reports/overnight/exp9_official.csv`
- Core Claim Fair Table: `reports/overnight/exp9_fair.csv`
- Core Claim Legacy v1: `reports/overnight/exp9_official_v1.csv`
- Conformal 5-Seed Coverage: `reports/conformal_coverage_seeds.csv`
- Split-Conformal Quantiles: `reports/conformal_quantiles.json`
- Uncertainty Evaluation: `reports/uncertainty_coverage.csv`
- GRU Checkpoint: `runs/gru_main/checkpoint.pt`
- Audit Status: `reports/overnight/audit_status.md`
- Run 4 Report: `reports/overnight/RUN4_REPORT.md`
