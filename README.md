# TyreSense: F1 Tyre Degradation Forecaster

TyreSense is an applied deep learning project that forecasts Formula 1 tyre degradation and remaining usable life using only public telemetry (`FastF1` and `OpenF1`).

Modern F1 race outcomes hinge on pit-stop strategy: determining when a tyre's pace falls off the **"degradation cliff"** ($+1.0\text{ s}$ slower than baseline for 2 consecutive laps from stint lap $\ge 4$). TyreSense mathematically isolates tyre degradation from fuel burn-off and trains sequence models (GRU, LSTM, RNN) to forecast lap times **20 laps ahead**.

---

## 🏎️ Key Features

- **Fuel Physics Correction**: Removes the speed gain caused by fuel consumption ($c = 0.03\text{ s/lap}$ scaled per circuit length):
  $$\text{lap\_time\_fc} = \text{lap\_time} - c \times (\text{total\_laps} - \text{lap\_number})$$
- **Causal Baseline Curves**: Quadratic wear curve fitted per `(track, compound)` on train split only (Audit C2) with causal early-laps offset.
- **Pure NumPy MLP**: From-scratch 8-input neural network with analytical gradient verification ($< 10^{-6}$) and backpropagation.
- **PyTorch Recurrent Forecasters**: GRU (5,626 params), LSTM (5,582 params), and RNN (5,726 params) with Huber loss and matched capacity ($\Delta \le 2.58\%$).
- **Interactive Stint Replay**: Streamlit app with season, circuit, session, driver, tyre compound selectors, and forward 20-lap trajectory forecast with responsive mobile view.
- **Leakage-Free Splitting & Locked Test Set**: 2022–2023 for training (38 races), 2024 for validation (21 races), and 2025 locked test split (20 races, R3 locked assertions).

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
│   ├── evaluate.py         # Multi-horizon MAE evaluation (h=1, 5, 10)
│   ├── mlp_numpy.py        # Pure NumPy MLP with analytical gradients & Adam
│   ├── models.py           # PyTorch GRUForecaster, LSTMForecaster, RNNForecaster
│   ├── train.py            # Custom training loop with Huber loss & early stopping
│   ├── claim_eval.py       # Core claim PWE and bootstrap comparison
│   ├── uncertainty.py      # MC-Dropout coverage calibration on real validation stints
│   └── runner_part_d.py    # Overnight PWE and cluster bootstrap evaluation
├── tests/                  # Pytest test suite (126 passing tests)
├── reports/                # Experimental outputs, baseline tables, and audit logs
└── notes/                  # Engineering rationales and viva notes
```

---

## 🚀 Quickstart

### 1. Installation & Environment
Recommended runtime: **Python 3.14.3** (`py -3.14` or system `python.exe`), which produced and verified all raw telemetry datasets. Python 3.10 is also supported for test execution.
Clone the repository and install dependencies:
```bash
git clone https://github.com/radheyashetty/TyreSense.git
cd TyreSense
pip install -r requirements.txt
```

### 2. Run Tests
Verify that all 126 unit, data invariant, and integration tests pass:
```bash
pytest
```

### 3. Launch Dashboard
Run the interactive Streamlit stint replay app:
```bash
streamlit run app.py
```

---

## 📊 Empirical Findings & Disclosures

1. **Core Scientific Claim (FROZEN vs. ROLLING)**:
   - On 2024 validation uncensored stints ($N=362$), ROLLING baseline curves decisively outperform FROZEN curves on lap-time forecasting across all horizons ($h=1, 5, 10$).
   - On remaining usable life (PWE), ROLLING gains an advantage of $+1.89\text{ laps}$ at Lap 5, $+3.60\text{ laps}$ at Lap 7, and $+5.16\text{ laps}$ at Lap 10 (both 95% stint bootstrap and race cluster bootstrap CIs strictly positive).
   - *Baseline Honesty Disclosure*: An Always-Train-Median baseline achieves a lower usable-life MAE ($\approx 3.9$ to $4.4\text{ laps}$) than polynomial baseline extrapolation ($7.6$ to $9.4\text{ laps}$), indicating polynomial curves struggle with non-linear accelerated wear.

2. **Model E Multiplier Inactivity**:
   - On baseline polynomial curves, $k$ does not update away from 1.0 because initial prediction error is mathematically zero when using causal baseline offsets ($g_t = \text{act}_t - \text{base}_t$). Model E behaves identically to ROLLING on smooth baseline curves.

3. **Uncertainty Calibration**:
   - Raw MC-dropout intervals are under-dispersed on real validation stints (raw 80% coverage: $11.9\%$, width: $1.46\text{ laps}$).
   - Calibrating with scale factor $s=4.0$ expands the 80% interval width to $5.85\text{ laps}$, achieving $20.4\%$ empirical coverage on real stints.

4. **Locked Test Set**:
   - Season 2025 data (20 races, 19,508 laps) remains completely locked and uninspected for all training, hyperparameter tuning, and metric reporting.

---

## 📜 Academic Integrity & Acknowledgments
- Telemetry provided by [FastF1](https://github.com/theOehrly/Fast-F1) and [OpenF1](https://openf1.org/).
- Circuit lengths verified against official FIA and Formula 1 specifications.
- This project is developed strictly for educational and scientific research purposes.
