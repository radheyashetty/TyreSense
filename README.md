# TyreSense: F1 Tyre Degradation Forecasting & Pit-Stop Window Estimation

TyreSense is an empirical deep learning system for Formula 1 tyre degradation modeling, multi-horizon lap time forecasting, and pit-stop window estimation. The project benchmarks deep recurrent neural networks (GRU, LSTM, Vanilla RNN) and a physics-guided adaptive model (Model E) across **91 Grand Prix races** from the 2022, 2023, 2024, and locked 2025 Formula 1 seasons (sourced via FastF1 and OpenF1 APIs).

---

## 1. Project Overview & Research Objectives

In modern Formula 1, race outcomes depend critically on tyre management and pit-stop timing. As tyres wear mechanically and thermally, lap times degrade until the tyre reaches a sharp performance drop-off known as the **tyre cliff**. 

The core research objectives of TyreSense are:
1. **Isolate Physical Tyre Degradation**: Remove confounding fuel mass consumption effects using circuit-calibrated physics corrections.
2. **Multi-Horizon Pace Forecasting**: Predict fuel-corrected lap times up to 20 laps into the future ($h = 1, \dots, 20$) given an observed stint history.
3. **Degradation Cliff Estimation**: Detect when fuel-corrected pace falls $\ge +1.0\text{ s}$ slower than the stint baseline for two consecutive qualifying laps.
4. **Principled Multi-Model Comparison**: Empirically evaluate GRU, LSTM, Vanilla RNN, physics-guided Model E, and heuristic baselines under matched capacity and strict temporal isolation.

---

## 2. Problem Statement & Motivation

During a Formula 1 Grand Prix, two competing physical phenomena occur simultaneously:
- **Fuel Burn-off**: Cars start with $\sim 110\text{ kg}$ of fuel, burning approximately $1.5\text{--}2.0\text{ kg}$ per lap. Lighter cars lap faster by approximately $0.03\text{ s/lap}$.
- **Tyre Degradation**: Chemical wear, graining, blistering, and thermal degradation reduce grip, making the car slower over time.

Because fuel burn-off masks tyre degradation in raw telemetry, a car's raw lap times may appear constant or faster even as its tyres wear dangerously close to the cliff. TyreSense solves this problem by applying a track-scaled fuel physics correction to unmask true tyre degradation and training sequence models on the unmasked pace residuals.

---

## 3. Architecture & Data Pipeline

```
Raw Telemetry (FastF1 & OpenF1: 91 Races)
                     │
                     ▼
       Cleaning & Quality Filters
   (Drop Lap 1, Pit Laps, SC/VSC, Inaccurate)
                     │
                     ▼
          Fuel Physics Correction
  lap_time_fc = lap_time - c_track * (total_laps - lap)
                     │
                     ▼
          Gap-Aware Stint Splitting
        (Split on telemetry gaps > 3 laps)
                     │
                     ▼
      Causal Baseline & Feature Extraction
  (Track/Compound Quadratic Curves + 8 Numeric Features)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Model Training            Online Dashboard
(GRU, LSTM, RNN, E)      (Streamlit + Replay)
```

### Data Preprocessing Steps:
1. **Basic Filtering (`src/clean.py`)**: Standing-start lap 1 is dropped; in-laps and out-laps (pit stops) are removed; Safety Car (SC) and Virtual Safety Car (VSC) laps are filtered out; deleted or telemetry-inaccurate laps are excluded.
2. **Fuel Physics Correction (`src/fuel.py`)**:
   $$\text{lap\_time\_fc} = \text{lap\_time} - c_{\text{track}} \times (\text{total\_laps} - \text{lap\_number})$$
   where $c_{\text{track}} = 0.03 \times (\text{circuit\_length} / 5.0\text{ km})$. Long circuits (e.g., Spa, $7.004\text{ km}$) burn more fuel per lap than short circuits (e.g., Monaco, $3.337\text{ km}$).
3. **Gap-Aware Stint Splitting (`src/clean.py`)**: Stints interrupted by multi-lap telemetry gaps (>3 laps) are split into separate continuous sub-stints.
4. **Causal Baseline Curve (`src/features.py`)**: A quadratic baseline curve $B(t) = a \cdot t^2 + b \cdot t + c$ is fitted per `(track_id, compound)` on the training split only.
5. **Feature Representation**: 8 numeric features (tyre age, lap fraction, relative pace, track temperature, track status, gap mask, track temperature delta, current residual) plus learned 4-dimensional track embeddings and 3-dimensional compound embeddings.

---

## 4. Deep Learning & Physics-Guided Models

TyreSense benchmarks four core approaches:

### A. GRU Forecaster (`src/models.py`)
- **Architecture**: 1-layer Gated Recurrent Unit (hidden dimension 32) with track and compound embeddings, projecting to a 20-step multi-horizon residual forecast.
- **Parameters**: 5,802 trainable parameters.
- **Mechanism**: Employs reset and update gates to regulate information flow without a separate cell state, offering strong representational capacity with fast convergence.

### B. LSTM Forecaster (`src/models.py`)
- **Architecture**: 1-layer Long Short-Term Memory network (hidden dimension 27) with track and compound embeddings.
- **Parameters**: 5,770 trainable parameters (matched within 1.6% capacity parity to GRU).
- **Mechanism**: Maintains an explicit cell state $c_t$ alongside hidden state $h_t$, regulated by input, forget, and output gates.

### C. Vanilla RNN Forecaster (`src/models.py`)
- **Architecture**: 1-layer Elman Recurrent Neural Network (hidden dimension 58) with track and compound embeddings.
- **Parameters**: 5,864 trainable parameters (matched within 1.6% capacity parity).
- **Mechanism**: Standard recurrent transition $h_t = \tanh(W x_t + U h_{t-1} + b)$. Susceptible to vanishing gradients over extended stint lengths.

### D. Model E — Physics-Guided Adaptive Forecaster (`src/live.py`)
- **Architecture**: Physics-guided online error-correction model without neural parameter optimization.
- **Mechanism**: Dynamically scales the causal quadratic baseline curve using an online wear multiplier $k$:
  $$k_{t+1} = \text{clip}\left(k_t + \eta \cdot \frac{g_t \cdot (y_t - \hat{y}_t)}{g_t^2 + 0.05},\, 0.5,\, 2.0\right)$$
  where $g_t$ is the baseline degradation gradient, $\eta = 0.3$ is the learning rate, and a 3-lap rolling average smooths rapid oscillations.

### E. Reference Baselines
- **Baseline Curve (Model A)**: Evaluates the static pre-race quadratic curve directly without online adaptation.
- **Persistence**: Predicts that all future laps will equal the last observed lap time ($\hat{y}_{t+h} = y_t$).

---

## 5. Training, Validation & Strict Test-Set Policy

### Temporal Data Split
To prevent chronological data leakage, races are partitioned strictly by season:
- **Training Split (2022–2023)**: 38 Grand Prix races. Used for curve fitting, embedding dictionary construction, and neural network training.
- **Validation Split (2024)**: 21 Grand Prix races. Used for model selection, hyperparameter tuning, conformal uncertainty calibration, and benchmark comparisons.
- **Locked Test Split (2025)**: 20 Grand Prix races. **Rule R3 strict lock**: Kept completely untouched, uninspected, and unevaluated until final submission.

### Training Protocol (`src/train.py`)
- **Loss Function**: Smooth Huber Loss ($\delta = 1.0$) to guard against telemetry outliers and traffic spikes:
  $$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta (|r| - \frac{1}{2} \delta) & \text{otherwise} \end{cases}$$
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Sequence Length**: 10 laps observation window, zero-padded for shorter early-stint histories.

---

## 6. Verified Empirical Benchmark Table & Scientific Analysis

All metrics below are verified directly from `reports/four_model_validation_summary.csv` evaluated across $N = 903$ stints from the 2024 validation dataset at cutoff lap 5 for a 20-lap horizon:

| Model | MAE $h=1$ (s) | MAE $h=5$ (s) | MAE $h=10$ (s) | Cliff PWE (laps) | $N$ Stints Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | **0.352** | **0.691** | **0.720** | 9.34 | 903 |
| **Model E (Adaptive)** | 0.367 | 0.798 | 1.013 | 7.17 | 903 |
| **GRU Forecaster** | 0.381 | 0.828 | 1.035 | **6.51** | 903 |
| **Vanilla RNN** | 0.381 | 0.848 | 1.152 | 6.73 | 903 |
| **LSTM Forecaster** | 0.381 | 0.861 | 1.125 | 6.62 | 903 |
| **Baseline Curve (Model A)** | 0.381 | 0.899 | 1.221 | 6.78 | 903 |

### Scientific Audit & Key Findings:

1. **Lap-Time MAE vs Persistence**:
   - Persistence achieves the lowest lap-time MAE at all three evaluated horizons ($0.352\text{ s}$ at $h=1$, $0.691\text{ s}$ at $h=5$, and $0.720\text{ s}$ at $h=10$).
   - *Why?* During the early and middle portions of an F1 stint (evaluated from cutoff lap 5), lap times change gradually lap-to-lap. Persistence acts as a low-variance predictor. Neural networks predict trend curvature, which introduces minor variance that slightly elevates point-wise lap time MAE.
   - *Academic Honesty Rule*: We do **not** claim that neural networks outperform persistence on short-to-medium horizon point-wise lap-time MAE on this dataset.

2. **Degradation Cliff Timing (PWE)**:
   - For strategic pit-stop decisions, the critical question is *when* the tyre will hit the degradation cliff, not just tracking flat lap times.
   - In cliff Point-Wise Error (PWE), **GRU achieves the lowest error (6.51 laps)**, followed by LSTM (6.62 laps), Vanilla RNN (6.73 laps), and Baseline Curve (6.78 laps).
   - Persistence exhibits the worst cliff timing error (**9.34 laps**), because a flat persistence line cannot anticipate non-linear tyre wear.
   - *Realistic Perspective*: While GRU is best among the tested approaches, an average cliff timing error of $\sim 6.5$ laps means forecasts provide an indicative window rather than lap-exact pit calls.

3. **Forecast Reconstruction Consistency**:
   - In continuity-adjusted mode, $\hat{y}[h] = y_{\text{last}} + \Delta B[h] + (\hat{r}[h] - \hat{r}[0])$. At $h=1$, the neural residual delta equals zero ($\hat{r}[0] - \hat{r}[0] = 0$), so GRU, LSTM, Vanilla RNN, and Baseline Curve all share the exact same step-1 forecast error ($0.381\text{ s}$).

---

## 7. Degradation Cliff Rule (Two Consecutive Laps)

A tyre degradation cliff is formally defined as:
$$\text{deg}_t \ge +1.0\text{ s} \quad \text{AND} \quad \text{deg}_{t+1} \ge +1.0\text{ s} \quad (\text{for } t \ge 4)$$
where $\text{deg}_t = t_{\text{fc}}[t] - t_{\text{baseline}}$, and $t_{\text{baseline}}$ is the driver's mean fuel-corrected lap time over stint laps 1–3.

- **Why Two Consecutive Laps?** Telemetry noise, driver lockups, defensive driving, and traffic frequently cause single-lap anomalies. Requiring two consecutive threshold crossings prevents false alarms.
- **Reporting Convention**: The cliff lap is designated as the **first lap** of the qualifying consecutive pair, matching the empirical ground-truth label in `src/clean.py`.

---

## 8. Installation & Environment Setup

### Prerequisites
- Python 3.10.x (recommended for PyTorch and Streamlit compatibility)
- PowerShell (Windows) or Bash (Linux/macOS)

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/your-username/TyreSense.git
cd TyreSense

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Launch Interactive Dashboard
```bash
streamlit run app.py
```

### Run Full Test Suite
```bash
# Default test suite (129 passed, 19 skipped in ~24s)
pytest

# Data invariant validation suite (18 passed in ~45s)
pytest -m data

# Randomized property and stability suite (1 passed in ~67s)
pytest -m slow

# Headless Streamlit application smoke test (10 passed in ~10s)
pytest tests/test_app.py
```

| Suite Name | Command | Results | Duration | Coverage & Verification Focus |
| :--- | :--- | :---: | :---: | :--- |
| **Default Suite** | `pytest` | **129 passed**, 19 skipped | $\sim 24\text{ s}$ | Core unit tests, sequence generation, model inference, cliff rules, and smoke tests. |
| **Data Invariants** | `pytest -m data` | **18 passed**, 130 deselected | $\sim 45\text{ s}$ | Raw telemetry schema invariants, time bounds, lap continuity, monotonic tyre age. |
| **Slow Property** | `pytest -m slow` | **1 passed**, 147 deselected | $\sim 67\text{ s}$ | Monte Carlo randomized property fuzzing across synthetic race distributions. |
| **Total Test Items** | — | **148 / 148 passed** | $\sim 136\text{ s}$ | **100% test pass rate**, 0 failures, 0 errors, 0 warnings. |

### Technical Reports & Audits
- **Final Technical Report**: [`reports/final_report.md`](file:///C:/Users/radhe/Desktop/deep%20learning/reports/final_report.md) (comprehensive summary of methodology and all experiments).
- **Dataset Utilization Audit**: [`reports/dataset_utilization_audit.md`](file:///C:/Users/radhe/Desktop/deep%20learning/reports/dataset_utilization_audit.md) (exact sequence accounting, target slot masking, and attrition analysis).
- **Four-Model Validation Benchmark**: [`reports/four_model_validation_summary.csv`](file:///C:/Users/radhe/Desktop/deep%20learning/reports/four_model_validation_summary.csv) (empirical evaluation across $N=903$ stints).
- **Viva Defense Cards & Task Log**: [`notes/explain.md`](file:///C:/Users/radhe/Desktop/deep%20learning/notes/explain.md) (engineering rationales and detailed defense concepts).


---

## 9. Dashboard Demonstration Procedure

To demonstrate TyreSense during an evaluation or viva presentation:

1. **Start the App**: Run `streamlit run app.py` and open the local browser URL (`http://localhost:8501`).
2. **Select Stint Telemetry**:
   - In the sidebar, select **Season**: `2024`.
   - Select **Circuit**: `Bahrain Grand Prix`.
   - Select **Driver**: `VER` (Max Verstappen).
   - Select **Tyre Type**: `SOFT` or `MEDIUM`.
   - Select **Stint**: Choose Stint 1.
3. **Step Through Stint History**:
   - Adjust the **Current Stint Lap** slider from Lap 1 to Lap 5.
   - Observe how the blue line represents observed laps and the orange line extends 20 laps into the future.
4. **Inspect Uncertainty & Cliff**:
   - Note the shaded orange band (80% split-conformal uncertainty envelope).
   - Check the red dashed horizontal line at $+1.0\text{ s}$ denoting the degradation cliff threshold.
5. **Switch to "Compare All Models"**:
   - In the sidebar radio button, select **Compare All Models**.
   - The main chart will simultaneously render:
     - **Blue**: Observed Telemetry
     - **Grey Dotted**: Frozen Lap 2 Baseline
     - **Orange**: GRU Forecaster
     - **Green Dashed**: LSTM Forecaster
     - **Purple Dotted**: Vanilla RNN
     - **Red Long-Dashed**: Model E (Physics-Guided Adaptive)
   - Inspect the summary table below the plot displaying each model's predicted cliff lap and remaining usable life.

---

## 10. Limitations, Technical Risks & Future Work

1. **No Counterfactual Compound Recommendations**:
   - TyreSense forecasts degradation for the compound currently on the car. It does **not** tell the team whether switching to Hard or Medium would be faster.
   - *Reason*: Telemetry provides observational data only for the compound actually driven. Counterfactual recommendation requires estimating unobserved tyre temperatures and grip levels under identical track rubbering, which cannot be proven without a full race simulator.
2. **Lap-Time MAE vs Pit Utility**:
   - Point-wise lap-time MAE in seconds is not a direct proxy for pit-stop quality. A model can have low MAE by predicting flat pace while completely missing the cliff.
3. **Sensor Noise & Unobserved Confounders**:
   - Real-world telemetry contains dirty air (wake from cars ahead), engine mode adjustments (ERS deployment), and traffic management that cannot be fully decoupled from physical tyre wear.
4. **Future Extensions**:
   - Integration of multi-car traffic simulators for counterfactual strategy evaluation.
   - Physics-informed neural networks (PINNs) coupling carcass thermal dynamics with telemetry pace.

---

## 11. Final-Year Viva Examination Defense Guide

### Q1: What problem does TyreSense solve?
**Answer**: TyreSense estimates tyre degradation and remaining tyre life in Formula 1 racing. By removing the confounding effect of fuel burn-off, it unmasks true tyre wear and forecasts future pace degradation to predict when a tyre will hit its degradation cliff.

### Q2: Why use recurrent neural networks for sequential lap data?
**Answer**: Lap telemetry is temporally correlated; tyre wear accumulates sequentially over time. Recurrent networks (GRU, LSTM, RNN) maintain an internal hidden state that summarizes historical pace and thermal degradation over previous laps, allowing multi-step forecasting without assuming lap independence.

### Q3: What is the architectural difference between GRU, LSTM, and Vanilla RNN?
**Answer**:
- **Vanilla RNN**: Uses a single $\tanh$ recurrent connection. It has no gating mechanism and suffers from vanishing gradients over sequences longer than 10–15 laps.
- **LSTM**: Uses three multiplicative gates (input, forget, output) and an additive cell state $c_t$ that allows gradients to flow over long sequences without exponential decay.
- **GRU**: Simplifies LSTM by merging cell and hidden states into a single vector $h_t$ and using two gates (reset and update), achieving comparable memory retention with fewer matrix operations.

### Q4: What is Model E and how does its physics-guided approach differ?
**Answer**: Model E does not train neural weights. It takes a pre-race quadratic baseline curve and updates an online wear multiplier $k$ using a normalized error-correction rule based on the difference between observed pace and baseline expectations. It adapts immediately to higher or lower track abrasion while bounding adaptation between $[0.5, 2.0]$.

### Q5: Why does persistence beat neural models at short horizons?
**Answer**: During early-to-mid stint laps, tyre degradation is gradual and lap times remain relatively steady. Persistence ($\hat{y}_{t+h} = y_t$) has zero variance from model misestimation. Neural networks attempt to predict non-linear curvature, which introduces slight variance on normal laps, giving persistence a lower point-wise MAE on this dataset.

### Q6: What do lap-time MAE and cliff Point-Wise Error (PWE) measure?
**Answer**: Lap-time MAE (in seconds) measures point-by-point pace tracking accuracy. Cliff PWE (in laps) measures error in predicting the specific lap when tyre degradation crosses the $+1.0\text{ s}$ threshold. A flat persistence model wins on MAE but fails completely on cliff PWE ($9.34$ laps error), whereas GRU achieves the lowest cliff PWE ($6.51$ laps).

### Q7: Why is a two-consecutive-lap threshold crossing used for cliff detection?
**Answer**: Telemetry often experiences single-lap spikes due to traffic, driver lockups, yellow flags, or overtaking maneuvers. Requiring degradation to exceed $+1.0\text{ s}$ for two consecutive laps filters out temporary disturbances and prevents premature, erroneous pit calls.

### Q8: What are data leakage and temporal validation?
**Answer**: Data leakage occurs when information from the future or from the test set influences model training. Standard k-fold cross-validation leaks information in time-series data because future laps would be used to predict past laps. Temporal validation partitions data strictly by chronology (2022–2023 for training, 2024 for validation, 2025 for testing), ensuring models only predict forward in time.

### Q9: Why is the 2025 test set kept strictly locked?
**Answer**: Under Rule R3, keeping the 2025 season completely untouched prevents iterative tuning bias, p-hacking, and test-set overfitting. It guarantees that final evaluation measures genuine out-of-sample generalizability across an unseen Formula 1 season.

### Q10: What are the main limitations of this system?
**Answer**: Key limitations include: (1) observational telemetry cannot isolate dirty air or engine mode changes from tyre wear; (2) cliff timing error remains $\sim 6.5$ laps on average; and (3) physical tyre wear cannot be directly measured, only inferred through lap-time proxies.

### Q11: Why does this project not yet recommend Soft, Medium, or Hard compounds?
**Answer**: Recommending an alternative compound is a counterfactual prediction. We observe telemetry only for the compound actually driven. Recommending a different tyre requires knowing how that compound would behave under identical track temperature, rubbering, and race traffic, which cannot be validated from historical race telemetry without a calibrated multi-agent race simulation.

### Q12: What evidence would be required before using these predictions for real-time pit strategy?
**Answer**: A race team would require: (1) validation on high-frequency ($10\text{ Hz}$) sensor data (wheel speed, tyre surface/carcass temperatures); (2) integration with traffic and pit-loss models ($20\text{--}25\text{ s}$ pit lane delta); (3) human-in-the-loop shadow testing during live race weekends; and (4) uncertainty intervals calibrated to safety-critical tolerances.
