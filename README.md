# TyreSense: F1 Tyre Degradation Forecaster

TyreSense is an applied deep learning project that forecasts Formula 1 tyre degradation and remaining usable life using only public telemetry (`FastF1` and `OpenF1`).

Modern F1 race outcomes hinge on pit-stop strategy: determining when a tyre's pace falls off the **"degradation cliff"** ($+1.0\text{ s}$ slower than baseline for 2 consecutive laps from stint lap $\ge 4$). TyreSense mathematically isolates tyre degradation from fuel burn-off and trains sequence models (GRU, LSTM, RNN) to forecast lap times **20 laps ahead**.

---

## 🏎️ Key Features

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
├── app.py                  # Streamlit interactive stint replay dashboard
├── config.yaml             # Central configuration (paths, seeds, horizons, models)
├── requirements.txt        # Pinned project dependencies
├── AGENT_RULES.md          # Architectural constraints (<=200 lines/file, <=30 lines/func)
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
│   ├── runner_part_c.py    # Multi-horizon comparison & baseline table generator
│   ├── runner_part_d.py    # EXP-9 paired PWE & cluster bootstrap evaluation
│   ├── runner_part_e.py    # Split-conformal calibration on 2024 validation races
│   ├── app_data.py         # App data loading and 2025 test lock enforcement
│   ├── app_forecast.py     # Multi-model forward forecasting engine
│   └── app_ui.py           # Plotly charts, responsive metric cards, and expanders
├── tests/                  # Pytest test suite (130 tests collected)
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
python -m src.runner_part_c

# Run EXP-9 Core Claim benchmark (FROZEN vs ROLLING vs Model E on 2024 validation)
python -m src.runner_part_d

# Run split-conformal uncertainty calibration on 2024 races
python -m src.runner_part_e
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```

### 4. Run Pytest Suite
```bash
# Standard test suite (111 passed, 19 skipped in ~21s)
pytest

# Data invariant validation suite (18 passed in ~35s)
pytest -m data

# Synthetic Monte Carlo property suite (1 passed in ~50s)
pytest -m slow
```

---

## 📊 Empirical Findings & Honest Disclosures (R1, R2)

1. **Multi-Horizon Forecast Comparison (`reports/baseline_table.csv`)**:
   - **Horizon $h=1$**: Simple pace persistence and ROLLING baseline offset achieve the lowest MAE ($0.653\text{ s}$), beating GRU ($0.947\text{ s}$), LSTM ($0.955\text{ s}$), and XGBoost ($0.955\text{ s}$).
   - **Horizons $h=5$ and $h=10$**: Recurrent neural networks decisively outperform all tabular and heuristic baselines (e.g., at $h=10$: LSTM $1.235\text{ s}$, GRU $1.248\text{ s}$ vs ROLLING $1.666\text{ s}$, Ridge $1.936\text{ s}$, XGBoost $1.958\text{ s}$).

2. **Core Claim Benchmark (`reports/overnight/exp9_results.csv`)**:
   - Dynamic ROLLING curve updates significantly outperform FROZEN curves on uncensored stints: ROLLING advantage of $+1.89\text{ laps}$ at Lap 5 (95% cluster CI $[1.26, 2.54]$) and $+5.16\text{ laps}$ at Lap 10 (95% cluster CI $[3.74, 6.43]$).
   - An Always-Train-Median heuristic achieves a lower point PWE ($\approx 3.9\text{--}4.4\text{ laps}$) by capturing the central distribution of compound lives.

3. **Model E (Online Wear Multiplier $k$)**:
   - Evaluated under the one-step-ahead update rule, $k$ dynamically adapts across **95.46%** of validation stints (median $k = 0.645$).
   - *Negative Result Disclosed*: During early laps (laps 3–7), tire management flattens telemetry wear relative to the quadratic curve ($\Delta \text{act} < \Delta \text{base}$), shifting $k < 1.0$ and initially increasing PWE before catching up at Lap 10 (PWE 6.89 vs 7.11).

4. **Split-Conformal Calibration (`reports/conformal_quantiles.json`)**:
   - Evaluated across 21 independent 2024 validation races (10 calibration races, 11 evaluation races).
   - Raw MC-dropout is severely underconfident (10.4% coverage at 80% nominal), whereas split-conformal calibration yields realistic bounds ($q = \pm 13.0\text{ laps}$ for 80% nominal coverage).

---

## 📌 Documented Code Exceptions & Stale Files

### AGENT_RULES.md Exceptions (Non-`nn.Module` Classes)
Per `AGENT_RULES.md`, plain functions are mandatory except for `torch.nn.Module`. Two documented architectural exceptions are maintained to preserve standard library integration and existing test contracts:
1. `src/train.py::StintSequenceDataset`: Subclass of `torch.utils.data.Dataset` required by PyTorch `DataLoader` for sequence batching.
2. `src/validate_raw.py::WarningHandler`: Subclass of `logging.Handler` required by the Python standard `logging` library to capture FastF1 warnings.

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
