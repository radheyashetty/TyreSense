# TyreSense 2025 Test Evaluation Protocol

**Status**: LOCKED (Awaiting Formal User Authorization)  
**Strict Mandate**: **DO NOT EXECUTE THIS PROTOCOL UNTIL FORMALLY UNLOCKED BY USER.**

---

## 1. Objective
Evaluate the frozen, pre-registered models on the held-out **2025 Formula 1 season** (20 non-wet races, 19,508 clean laps) to measure true out-of-sample generalization without any prior exposure, tuning, or leakage.

---

## 2. Checkpoints & Pre-Registered Configurations
The test evaluation must strictly evaluate the following pre-registered model weights:
- **Primary Neural Model**: `runs/gru_main/checkpoint.pt` (GRU, 5,802 params, trained on 2022–2023 with seed 42)
- **Model Configurations**:
  - `runs/gru_main/config.json`
  - `runs/lstm_main/config.json`
  - `runs/rnn_main/config.json`
  - `runs/xgboost/config.json`
  - `runs/ridge/config.json`
- **Baseline Wear Curves**: Curves fitted strictly on 2022–2023 training races (`curves` and `global_curve`).

---

## 3. Pre-Registered Conformal Quantiles
Uncertainty intervals will evaluate the pre-fitted split-conformal quantiles from `reports/conformal_quantiles.json`:
- Nominal 50%: $q = \pm 7.0\text{ laps}$
- Nominal 60%: $q = \pm 9.0\text{ laps}$
- Nominal 70%: $q = \pm 10.0\text{ laps}$
- Nominal 80%: $q = \pm 13.0\text{ laps}$
- Nominal 90%: $q = \pm 18.0\text{ laps}$

---

## 4. Evaluation Metrics
1. **Multi-Horizon Lap Time MAE**: Measured at $h = 1, 5, 10$ laps on 2025 test sliding windows for GRU, LSTM, RNN, XGBoost, Ridge, and persistence.
2. **EXP-9 Core Claim PWE**: Standardized Pit Window Error $|\min(R_{\text{pred}}, 20) - \min(R_{\text{true}}, 20)|$ measured on uncensored 2025 stints at evaluation laps 3, 5, 7, 10.
3. **Conformal Empirical Coverage**: Measured on uncensored 2025 stints against the 5 nominal coverage levels.

---

## 5. Single Execution Command (When Formally Authorized)
```bash
python -m src.final_eval --checkpoint runs/gru_main/checkpoint.pt --quantiles reports/conformal_quantiles.json --split test --output reports/final_2025_test_results.csv
```
*(DO NOT RUN NOW - R3 TEST LOCK ACTIVE)*
