# TyreSense: F1 Pit-Stop Window Forecasting via Deep Recurrent Networks

TyreSense is an empirical deep learning system for Formula 1 tire degradation modeling and pit-stop window forecasting. It processes **91 Grand Prix races** across the 2022, 2023, 2024, and locked 2025 hybrid era seasons (sourced from FastF1 and OpenF1 APIs).

---

## 🏎️ Core Scientific Architecture

Modern F1 race outcomes hinge on pit-stop strategy: determining when a tyre's pace degrades past a critical threshold (the "tyre cliff", defined here as $+1.0\text{ s}$ over baseline). TyreSense evaluates heuristic baselines, tree regressors, and deep recurrent neural networks against rigorous temporal splits.

- **Fuel Physics Correction**: Removes the speed gain caused by fuel consumption ($c = 0.03\text{ s/lap}$ scaled per circuit length):
  $$\text{lap\_time\_fc} = \text{lap\_time} - c \times (\text{total\_laps} - \text{lap\_number})$$
- **Causal Baseline Curves**: Quadratic wear curve fitted per `(track, compound)` on train split only with causal early-laps offset.
- **Pure NumPy MLP**: From-scratch 8-input neural network with analytical gradient verification ($< 10^{-6}$) and backpropagation.
- **PyTorch Recurrent Forecasters**: GRU (5,802 params), LSTM (5,770 params), and RNN (5,864 params) with Huber loss and capacity parity ($\Delta \le 1.6\%$).
- **Multi-Horizon Forecaster**: Direct multi-step ahead prediction ($h=1, 5, 10$) evaluated across 3 independent seeds.
- **Split-Conformal Usable Life Intervals**: Finite-sample calibrated prediction intervals on 2024 validation races.
- **Interactive Stint Replay**: Streamlit app with season, circuit, session, driver, tyre compound selectors, model switcher, and responsive view.
- **Strict Test Lock**: 2022–2023 for training (38 races), 2024 for validation (21 races), and 2025 locked test split (20 races, R3 locked assertions).

---

## 📁 Repository Structure

```
TyreSense/
├── data/
│   ├── raw/                # 91 raw Parquet race telemetry files (FastF1 & OpenF1)
│   ├── raw_excluded/       # 1 excluded wet race (Suzuka 2022)
│   └── processed/          # Cleaned & fuel-corrected train, val, and locked splits
├── src/                    # Core source modules
│   ├── download.py         # OpenF1/FastF1 telemetry ingestion & parquet export
│   ├── clean.py            # Quality filters, attrition accounting & persistent cliff labels
│   ├── fuel.py             # Circuit-scaled fuel slope calculation
│   ├── split.py            # Temporal splitting (2022-2023 / 2024 / 2025 lock)
│   ├── features.py         # Causal baseline curves & 8-feature sequence builder
│   ├── baselines.py        # Model A (Curve), Model B (Ridge), Model C (XGBoost)
│   ├── evaluate.py         # Baseline evaluation interface
│   ├── mlp_numpy.py        # Pure NumPy MLP with analytical gradients & Adam
│   ├── models.py           # PyTorch GRUForecaster, LSTMForecaster, RNNForecaster
│   ├── train.py            # Training loop with Huber loss & early stopping
│   ├── live.py             # Online error-correction wear multiplier k
│   ├── claim_eval.py       # Core claim PWE and bootstrap comparison
│   ├── eval_baselines.py   # Multi-horizon comparison & baseline table generator
│   ├── eval_pwe.py         # EXP-9 standardized PWE & cluster bootstrap evaluation
│   ├── eval_uncertainty.py # Split-conformal calibration on 2024 validation races
│   ├── app_data.py         # App data loading and 2025 test lock enforcement
│   ├── app_forecast.py     # Multi-model forward forecasting engine
│   └── app_ui.py           # Plotly charts, responsive metric cards, and expanders
├── tests/                  # Pytest test suite (131 tests collected)
├── reports/                # Experimental outputs, baseline tables, and audit logs
└── notes/                  # Engineering rationales and viva notes
```

---

## 🚀 Execution & Reproduction Commands

### 1. Ingestion & Preprocessing
```bash
# Ingest raw telemetry (FastF1 for 2022, OpenF1 for 2023-2024)
python -m src.download --season 2022

# Clean telemetry, apply circuit fuel corrections, and assign cliff labels
python -m src.clean
```

### 2. Model Training & Comparison
```bash
# Train main GRU neural forecaster
python -m src.train --epochs 25 --run-name gru_main --seed 42

# Run full baseline & recurrent comparison (GRU, LSTM, RNN across 3 seeds; XGBoost, Ridge, Baselines)
python -m src.eval_baselines

# Run EXP-9 Core Claim benchmark (FROZEN vs ROLLING vs Model E on 2024 validation)
python -m src.eval_pwe

# Run split-conformal uncertainty calibration on 2024 races
python -m src.eval_uncertainty
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```

### 4. Run Pytest Suite
```bash
# Standard test suite (112 passed, 19 skipped in ~22s)
pytest

# Data invariant validation suite (18 passed in ~35s)
pytest -m data

# Synthetic Monte Carlo property suite (1 passed in ~50s)
pytest -m slow
```

---

## 📊 Empirical Findings & Honest Disclosures (R1, R2)

1. **Multi-Horizon Forecast Comparison (`reports/baseline_table.csv`)**:
   - **Horizon $h=1$**: Simple pace persistence achieves the lowest MAE ($0.653\text{ s}$), beating GRU ($0.947\text{ s}$), LSTM ($0.955\text{ s}$), and XGBoost ($0.955\text{ s}$).
   - **Horizons $h=5$ and $h=10$**: Recurrent neural networks outperform tabular and heuristic baselines (e.g., at $h=10$: LSTM $1.235\text{ s}$, GRU $1.248\text{ s}$ vs persistence $1.666\text{ s}$, Ridge $1.936\text{ s}$, XGBoost $1.958\text{ s}$).

2. **Core Claim Benchmark (`reports/overnight/exp9_official.csv`)**:
   - Standardized PWE evaluates $N=345$ uncensored stints (clean length $\ge 10$) across all laps 3, 5, 7, 10.
   - Dynamic ROLLING curve updates outperform FROZEN curves on uncensored stints: ROLLING advantage of $+1.62\text{ laps}$ at Lap 5 (95% cluster CI $[0.86, 2.37]$) and $+4.99\text{ laps}$ at Lap 10 (95% cluster CI $[3.56, 6.27]$). At Lap 3, advantage is $+0.14\text{ laps}$ with 95% cluster CI $[-0.14, 0.44]$ crossing zero.
   - A naive Always-Train-Median heuristic achieves a lower point PWE ($\approx 4.2\text{--}4.9\text{ laps}$) by capturing the central distribution of compound lives.

3. **Model E (Online Wear Multiplier $k$)**:
   - Evaluated under the one-step-ahead update rule across 2024 validation stints.
   - *Negative Result Disclosed*: During early laps (laps 2–8), telemetry degradation deltas are flatter than the quadratic curve, causing $k$ to collapse to the lower clamp ($k=0.5$) on 27.8% of validation stints. Consequently, Model E underperforms plain ROLLING (by $-0.57\text{ laps}$ at Lap 3, $-1.51$ at Lap 5, $-1.96$ at Lap 7, $-3.38$ at Lap 10).

4. **Split-Conformal Calibration (`reports/conformal_quantiles.json`)**:
   - Evaluated across 21 independent 2024 validation races (10 calibration races, 11 evaluation races).
   - Raw MC-dropout is severely underconfident (10.4% coverage at 80% nominal), whereas split-conformal calibration yields realistic bounds ($q = \pm 13.0\text{ laps}$ for 80% nominal coverage, with 66.3% measured coverage on evaluation races).

---

## 📌 Documented Code Exceptions & Stale Files

### AGENT_RULES.md Exceptions (Non-`nn.Module` Classes)
Per `AGENT_RULES.md`, plain functions are mandatory except for `torch.nn.Module`. Two documented architectural exceptions are maintained to preserve standard library integration and existing test contracts:
1. `src/train.py::StintSequenceDataset`: Subclass of `torch.utils.data.Dataset` required by PyTorch `DataLoader` for sequence batching.
2. `src/validate_raw.py::WarningHandler`: Subclass of `logging.Handler` required by the Python standard `logging` library to capture FastF1 warnings during automated extraction without process termination.

### Stale Artifacts (Preserved for Provenance, Not Used)
The following files are preserved for historical audit provenance and are never loaded by the current pipeline:
- `reports/races_old_part1.csv`
- `reports/part1_old/`
- `runs/invalid_final_2023_02/`

---

## 📜 Academic Integrity & Telemetry Acknowledgments
- Telemetry provided by [FastF1](https://github.com/theOehrly/Fast-F1) and [OpenF1](https://openf1.org/).
- Circuit lengths verified against official FIA and Formula 1 specifications.
- Developed strictly for educational and scientific research purposes.
