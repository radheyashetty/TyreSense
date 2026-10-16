# TyreSense: F1 Tyre Degradation Forecaster
## Comprehensive Project Report: Current Architecture, Part 2 Roadmap, & Engineering Recommendations

---

## 1. Executive Summary & Vision

**TyreSense** is an applied deep learning project developed to forecast Formula 1 tyre degradation and remaining usable life using only public telemetry (`FastF1`). 

In modern Formula 1, race outcomes hinge on pit stop strategy: determining when a tyre's pace falls off the **"degradation cliff"** ($+1.0\text{ s}$ slower than baseline). TyreSense isolates tyre degradation from the confounding effect of fuel burn-off and trains sequence models (GRU, LSTM) to forecast lap times **20 laps ahead**.

The project is structured in two distinct phases:
- **Part 1 (Completed & Verified)**: Clean dataset construction, physics-based fuel correction, causal baseline curves, pure NumPy MLP backpropagation, PyTorch deep neural models (GRU, LSTM, RNN), 4 ablation experiments, and an interactive Streamlit replay dashboard.
- **Part 2 (Upcoming)**: Testing the core empirical claim: *Does a live-updating forecast (Rolling mode) beat a static forecast issued at lap 2 (Frozen mode)?*, plus live adaptive Bayesian correction ($k$-factor), MC-Dropout uncertainty bands, and pit-window computation.

---

## 2. What the Project Currently Does (Part 1 Breakdown)

```
[FastF1 Raw Telemetry] 
       │
       ▼
[src/clean.py: Filters & Fuel Correction] ──► reports/attrition.csv
       │
       ▼
[src/split.py: Season Split (2022-23 Train / 2024 Val / 2025 Locked)]
       │
       ├──► [src/baselines.py: Model A (Curve), Model B (Ridge), Model C (XGBoost)]
       │
       ├──► [src/mlp_numpy.py: 7-16-8-1 MLP, Analytical Gradients & Adam]
       │
       └──► [src/models.py & src/train.py: PyTorch GRU / LSTM / RNN Forecasters]
                   │
                   ▼
       [reports/ablation.csv & runs/pilot/metrics.json]
                   │
                   ▼
       [app.py: Interactive Stint Replay & 20-Lap Forward Predictor]
```

### Key Modules Implemented in Part 1

1. **Telemetry Ingestion & Caching (`src/download.py`)**:
   - Integrates with `FastF1`, downloads full race sessions, and stores them in high-speed parquet format (`data/raw/`).
   - Supports zero-network local loading from cache.

2. **Domain Cleaning & Fuel Physics (`src/clean.py`, `src/fuel.py`)**:
   - **Fuel Normalization**: Strips the natural speed gain caused by fuel consumption ($c = 0.03\text{ s/lap}$) using:
     $$\text{lap\_time\_fc} = \text{lap\_time} - 0.03 \times (\text{total\_laps} - \text{lap\_number})$$
   - **Quality Attrition**: Discards wet-weather races, lap 1 chaos, in/out laps, and SC/VSC disruptions. Reconciles every dropped row into `reports/attrition.csv`.
   - **Stint Labeling**: Computes baseline pace from laps 1–3 and labels the usable-life cliff when degradation reaches $+1.0\text{ s}$. Right-censors stints where the tyre never hit the cliff.

3. **Leakage-Free Splitting (`src/split.py`)**:
   - Enforces temporal integrity: **2022–2023** for training, **2024** for validation, and **2025** locked for the final evaluation.
   - Scalers and track vocabularies fit strictly on training splits; unseen tracks map to a reserved `"unk"` token.

4. **Causal Quadratic Baseline Curves (`src/features.py`)**:
   - Fits causal quadratic wear curves per `(track, compound)`.
   - Before stint lap 3, uses the track/compound curve; from lap 3 onwards, offsets the curve to match the driver's observed pace level.
   - Generates 7 normalized inputs and a 20-step forward residual target vector.

5. **Competitive Baselines (`src/baselines.py`, `src/evaluate.py`)**:
   - **Model A**: Causal quadratic curve.
   - **Model B**: Ridge regression.
   - **Model C**: Gradient-boosted decision trees (XGBoost).
   - Generates multi-horizon MAE evaluation ($h = 1, 5, 10$) in `reports/baseline_table.csv`.

6. **Pure NumPy MLP with Analytical Gradients (`src/mlp_numpy.py`)**:
   - Implements a 7-16-8-1 neural network from scratch using pure NumPy matrix operations.
   - Backpropagation verified against numerical two-sided gradients:
     $$\frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytical}}\| + \|\nabla_{\text{numerical}}\|} = 1.42 \times 10^{-8} \ll 10^{-6}$$
   - Verified exact equivalence with PyTorch Adam optimizer ($0.00\%$ loss discrepancy).

7. **Deep Sequence Forecasters (`src/models.py`, `src/train.py`)**:
   - PyTorch `nn.Module` forecasters: **GRU** (5,626 parameters), **LSTM** (7,162 parameters), and **Vanilla RNN** (2,554 parameters).
   - Custom CPU training loop featuring Huber loss ($\delta = 1.0$), gradient norm clipping ($1.0$), and `ReduceLROnPlateau`.

8. **Ablation Studies (`src/ablation.py`)**:
   - Conducted and saved EXP-2 (activations & losses), EXP-3 (optimizers & batch sizes), EXP-6 (static baselines), and EXP-7 (recurrent architectures) to `reports/ablation.csv`.

9. **Interactive Streamlit Replay Dashboard (`app.py`)**:
   - Interactive replay of historic Grand Prix stints from Bahrain and Saudi Arabia.
   - Circuit, Session, Driver, Tyre Compound, and Stint selectors.
   - Interactive lap-by-lap scrubbing with Forward/Backward buttons.
   - Plotly visualization showing observed pace vs. the 20-lap predicted forward trajectory and $+1.0\text{ s}$ cliff line.
   - Real-time telemetry cards: Circuit, Driver/Tyre, Stint Lap, Estimated Live Fuel on Board ($110\text{ kg} \to 5\text{ kg}$), and Remaining Usable Life.

---

## 3. What the Project Will Do in Part 2

Part 2 answers the central scientific question of the project: **Does continuous real-time telemetry updating actually beat a static prediction made at lap 2?**

```
Part 2 Workflow:
┌────────────────────────────────────────────────────────┐
│  Lap 2 Static (FROZEN)  vs.  Live Lap-by-Lap (ROLLING) │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
  [src/live.py: k-factor]    [src/uncertainty.py]
  k_new = k + eta * g * err  MC-Dropout (30 passes)
  Adaptive Wear Rate Multiplier   P10, P50, P90 Intervals
             │                           │
             └─────────────┬─────────────┘
                           ▼
             [Pit Window: [P10 - 2, P50]]
                           ▼
          [app.py Extended Dashboard]
```

### Specific Part 2 Deliverables:
1. **Inference Modes (`evaluate.py`)**:
   - **FROZEN Mode**: Prediction issued once at stint lap 2 and frozen.
   - **ROLLING Mode**: Prediction recalculated at every single lap using newly observed telemetry.
2. **Online Adaptive Correction (`src/live.py`)**:
   - Tracks a real-time tyre degradation multiplier $k \in [0.5, 2.0]$.
   - If a driver pushes aggressively or the track is more abrasive, $k$ adapts upward ($k > 1.0$) using error-normalized gradient updates.
3. **Uncertainty Quantification (`src/uncertainty.py`)**:
   - Runs 30-pass Monte Carlo Dropout during inference.
   - Computes $P_{10}$, $P_{50}$, and $P_{90}$ confidence intervals for remaining tyre life.
   - Calibrates temperature scaling on training folds to guarantee that the 80% confidence band covers 75–85% of real race stints.
4. **Pit Window Formulation**:
   - Automatically computes the strategic pit stop window:
     $$\text{Pit Window} = [\text{current\_lap} + P_{10} - 2,\; \text{current\_lap} + P_{50}]$$
5. **Ablations & Sensitivity Analysis**:
   - **EXP-8**: Gradient flow & norm comparison across recurrent architectures.
   - **EXP-9**: Empirical claim test: Frozen vs. Rolling vs. Live-Corrected.
   - **EXP-10**: Coverage validation of uncertainty intervals.
   - Sensitivity checks on fuel burn slope ($0.02, 0.03, 0.04\text{ s/lap}$) and cliff definition ($0.5, 1.0, 1.5\text{ s}$).
6. **2025 Test Set Opening & Error Analysis**:
   - Evaluates the frozen model once on the unseen 2025 championship data.
   - Deep-dive error analysis on the 10 worst predicted stints.

---

## 4. Current File Structure & Inventory

```
deep learning/
│
├── .gitignore                      # Excludes raw data, caches, and checkpoints
├── AGENT_RULES.md                  # Strict guidelines (<=200 lines/file, <=30 lines/func)
├── config.yaml                     # Centralized parameters, seeds, paths, horizons
├── PART1_PRD.md                    # Specifications for Tasks T1.1 - T1.9 (Completed)
├── PART2_PRD.md                    # Specifications for Tasks T2.1 - T2.10 (Upcoming)
├── requirements.txt                # Pinned dependencies (fastf1, torch, xgboost, streamlit)
├── app.py                          # Streamlit interactive stint replay dashboard
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Parquet files downloaded from FastF1
│   │   ├── race_2023_01.parquet    # Bahrain GP raw data
│   │   └── race_2023_02.parquet    # Saudi Arabian GP raw data
│   └── processed/                  # Cleaned, fuel-corrected analytical data
│       ├── laps.parquet            # 1,663 clean race laps across Bahrain & Jeddah
│       └── stints.parquet          # 98 clean stints (>= 8 laps)
│
├── fastf1_cache/                   # On-disk HTTP & timing cache for FastF1
│
├── notes/
│   └── explain.md                  # Detailed task-by-task engineering rationale & notes
│
├── reports/                        # Verification outputs & experimental records
│   ├── ablation.csv                # Results from EXP-2, EXP-3, EXP-6, and EXP-7
│   ├── attrition.csv               # Data cleaning filter accounting & drop counts
│   ├── baseline_table.csv          # Multi-horizon MAE for Models A, B, and C
│   └── project_overview.md         # This report
│
├── runs/
│   └── pilot/
│       └── metrics.json            # Baseline metrics from initial GRU pilot training run
│
├── src/                            # Production source code
│   ├── __init__.py                 # Package marker
│   ├── ablation.py                 # Runner for EXP-2, 3, 6, 7
│   ├── baselines.py                # Implementations of Model A (Curve), B (Ridge), C (XGBoost)
│   ├── clean.py                    # Multi-step filtering, censoring, and stint generation
│   ├── config.py                   # Configuration loader
│   ├── download.py                 # FastF1 automated downloader and extractor
│   ├── evaluate.py                 # Multi-horizon MAE and censoring metric calculator
│   ├── features.py                 # Quadratic baseline curves and 7-input sequence builder
│   ├── fuel.py                     # Deterministic fuel penalty adjustment logic
│   ├── mlp_numpy.py                # Pure NumPy 7-16-8-1 MLP with analytical gradients & Adam
│   ├── models.py                   # PyTorch GRUForecaster, RNNForecaster, LSTMForecaster
│   ├── split.py                    # Leakage-free temporal splitter & vocabulary encoder
│   └── train.py                    # Hand-written CPU training loop with Huber loss
│
└── tests/                          # Automated pytest suite (23 / 23 passing)
    ├── test_ablation.py            # Verifies EXP-2/3/6/7 table generation
    ├── test_app.py                 # Tests app latency (<3s first load, <1s step) and causality
    ├── test_baselines.py           # Validates Models A, B, C and evaluation tables
    ├── test_clean.py               # Tests fuel removal, filter counts, and attrition reconciliation
    ├── test_download.py            # Verifies offline loading from raw parquet cache
    ├── test_features.py            # Tests causal curve fitting and sequence creation
    ├── test_mlp_numpy.py           # Verifies analytical gradient check (<1e-6) and PyTorch parity
    ├── test_models.py              # Validates PyTorch model parameter counts and forward shapes
    └── test_split.py               # Enforces temporal separation and unk token handling
```

---

## 5. Current State of the Codebase

- **Test Suite Status**: **23 out of 23 tests passing** via `pytest`.
- **Latency Benchmarks**:
  - `app.py` initial data load from cache: **0.099 s** (Target: $< 3.0\text{ s}$).
  - Single-step causal forecast computation: **0.117 s** (Target: $< 1.0\text{ s}$).
- **Architectural Constraints (`AGENT_RULES.md`)**:
  - Every file in `src/` and `app.py` is under 200 lines.
  - Every function is $\le 30$ lines.
  - No lambdas, no prohibited decorators, plain and readable code.

---

## 6. Recommendations & High-Impact Suggestions

Based on industry research (Mercedes-AMG / Imperial College London SAC 2025) and top GitHub implementations, here are high-impact enhancements to consider:

1. **Track-Length Scaled Fuel Penalty**:
   Instead of a static $c = 0.03\text{ s/lap}$, scale $c$ by track distance:
   $$c_{\text{track}} = 0.03 \times \frac{\text{Circuit Length (km)}}{5.0\text{ km}}$$
   This prevents under-correcting long power circuits (Spa, Monza) and over-correcting short tracks (Monaco).

2. **Single-Lap Traffic & Lockup Outlier Filtering**:
   If a single lap drops by $> 2.0\text{ s}$ and the following lap immediately recovers to normal pace, flag it as a traffic or lockup anomaly rather than permanent tyre degradation.

3. **Track Temperature Delta ($\Delta T_{\text{track}}$)**:
   In twilight races like Bahrain or Abu Dhabi, asphalt temps drop by up to $12^\circ\text{C}$. Introducing $\Delta T = T_{\text{current}} - T_{\text{stint\_start}}$ captures night-time thermal grip recovery.

4. **Monotonicity Parameterization in the 20-Lap Output Head**:
   Parameterizing the forward prediction as cumulative positive degradation steps ($\hat{g}_{t+h} = \hat{g}_t + \sum \text{ReLU}(\Delta g)$) guarantees the network will never predict non-physical "spontaneous tyre rejuvenation".

5. **Pit Window Error (PWE) Evaluation Metric**:
   In addition to lap-time MAE (seconds), evaluate **Pit Window Accuracy** in laps:
   $$\text{PWE} = |\text{Predicted Cliff Lap} - \text{Actual Cliff Lap}|$$
   This matches the primary KPI used by real-world F1 strategists.
