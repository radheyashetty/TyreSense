# TyreSense Measured Facts Ledger

**Generated**: 2026-10-21T09:27:00+05:30  
**Branch**: `overnight-run`  

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
- **Locked Test Split (2025)**: 19,508 clean laps (20 non-wet races, held uninspected)

---

## 3. Data Cleaning Pipeline (Code Order)
From `src/clean.py`:
1. **Wet-Race Exclusion**: Drop entire races where $> 10\%$ of laps have `Rainfall == True` or compound is `INTERMEDIATE` / `WET`.
2. **Missing Lap Time Removal**: Filter out rows with null `LapTime`.
3. **Compound Filtering**: Keep only dry slicers (`SOFT`, `MEDIUM`, `HARD`); drop `None`, null, or unknown compounds.
4. **IsAccurate Telemetry Filter**: For FastF1 (2022), drop laps where `IsAccurate == False`; for OpenF1 (2023–2025) where `IsAccurate` is NaN, retain laps if non-outlier.
5. **Physical Speed Filter**: Retain only laps with $50.0\text{ s} \le \text{LapTime} \le 150.0\text{ s}$.
6. **Circuit-Scaled Fuel Physics Correction**:
   $$c = 0.03 \times \frac{\text{Circuit Length (km)}}{5.303\text{ km}}$$
   $$\text{lap\_time\_fc} = \text{lap\_time} - c \times (\text{total\_laps} - \text{lap\_number})$$
7. **Causal Spike Flagging**: Flag laps $> 1.5\text{ s}$ slower than recent 3-lap rolling median as safety car / traffic anomalies.

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

## 6. Output Artifact Paths
- Baseline Comparison: `reports/baseline_table.csv`
- Core Claim EXP-9: `reports/overnight/exp9_results.csv`
- Split-Conformal Quantiles: `reports/conformal_quantiles.json`
- Uncertainty Evaluation: `reports/uncertainty_coverage.csv`
- Final Run Report: `reports/overnight/FINAL_RUN_REPORT.md`
- GRU Checkpoint: `runs/gru_main/checkpoint.pt`
- Model Configurations: `runs/gru_main/config.json`, `runs/lstm_main/config.json`, `runs/rnn_main/config.json`, `runs/xgboost/config.json`, `runs/ridge/config.json`
