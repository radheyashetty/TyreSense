# TyreSense: F1 Tyre Degradation Forecaster

TyreSense is an applied deep learning project that forecasts Formula 1 tyre degradation and remaining usable life using only public telemetry (`FastF1`).

Modern F1 race outcomes hinge on pit-stop strategy: determining when a tyre's pace falls off the **"degradation cliff"** ($+1.0\text{ s}$ slower than baseline). TyreSense mathematically isolates tyre degradation from fuel burn-off and trains sequence models (GRU, LSTM) to forecast lap times **20 laps ahead**.

---

## 🏎️ Key Features

- **Fuel Physics Correction**: Removes the speed gain caused by fuel consumption ($c = 0.03\text{ s/lap}$) using:
  $$\text{lap\_time\_fc} = \text{lap\_time} - 0.03 \times (\text{total\_laps} - \text{lap\_number})$$
- **Causal Baseline Curves**: Quadratic wear curve fitted per `(track, compound)` with stint-level offset from lap 3.
- **Pure NumPy MLP**: From-scratch 7-16-8-1 neural network with backpropagation and analytical gradient verification ($1.42 \times 10^{-8} < 10^{-6}$).
- **PyTorch Recurrent Forecasters**: GRU (5,626 params), LSTM (7,162 params), and RNN (2,554 params) with Huber loss.
- **Interactive Stint Replay**: Streamlit app with circuit, session, driver, tyre compound selectors, and forward 20-lap trajectory forecast.
- **Leakage-Free Splitting**: 2022–2023 for training, 2024 for validation, and 2025 locked test split.

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit interactive stint replay dashboard
├── config.yaml             # Central configuration (paths, seeds, horizons, models)
├── requirements.txt        # Pinned project dependencies
├── AGENT_RULES.md          # Architectural constraints (<=200 lines/file, <=30 lines/func)
├── PART1_PRD.md            # Part 1 task specifications (T1.1 - T1.9: Completed)
├── PART2_PRD.md            # Part 2 task specifications (T2.1 - T2.14: In progress)
├── src/                    # Core source modules
│   ├── download.py         # FastF1 telemetry ingestion & parquet export
│   ├── clean.py            # Quality filters, attrition accounting & usable-life labels
│   ├── fuel.py             # Fuel normalization slope calculation
│   ├── split.py            # Season splitting & vocabulary encoding
│   ├── features.py         # Causal baseline curves & 7-input sequence builder
│   ├── baselines.py        # Model A (Curve), Model B (Ridge), Model C (XGBoost)
│   ├── evaluate.py         # Multi-horizon MAE evaluation (h=1, 5, 10)
│   ├── mlp_numpy.py        # Pure NumPy MLP with analytical gradients & Adam
│   ├── models.py           # PyTorch GRUForecaster, LSTMForecaster, RNNForecaster
│   ├── train.py            # Custom training loop with Huber loss
│   └── ablation.py         # EXP-2, EXP-3, EXP-6, EXP-7 experiment runner
├── tests/                  # Pytest test suite (23 / 23 passing)
├── reports/                # Experimental outputs, attrition, and baseline tables
└── notes/                  # Engineering rationales and viva notes
```

---

## 🚀 Quickstart

### 1. Installation & Environment
Recommended runtime: **Python 3.14.3** (`py -3.14` or system `python.exe`), which produced and verified all raw telemetry datasets. Python 3.10 is also supported for test execution.
Clone the repository and install the dependencies:
```bash
git clone https://github.com/radheyashetty/TyreSense.git
cd TyreSense
pip install -r requirements.txt
```

### 2. Run Tests
Verify that all unit and integration tests pass:
```bash
pytest tests/
```

### 3. Launch the Dashboard
Run the interactive Streamlit stint replay app:
```bash
streamlit run app.py
```

---

## 📊 Status & Roadmap

- **Part 1 (Completed)**: Data pipeline, fuel physics, causal quadratic curves, pure NumPy MLP, GRU/LSTM forecasters, ablation suites (EXP-2, 3, 6, 7), and the interactive Streamlit dashboard. All 23 tests passing.
- **Part 2 (In Progress)**:
  - Testing the core claim: **FROZEN vs. ROLLING inference**.
  - Online adaptive wear correction ($k$-factor).
  - 30-pass MC-Dropout uncertainty bands and pit window calculation ($[P_{10} - 2, P_{50}]$).
  - Opening the locked 2025 test season.

---

## 📜 License & Acknowledgments
Built with telemetry from [FastF1](https://github.com/theOehrly/Fast-F1). This project is developed strictly for educational and research purposes.
