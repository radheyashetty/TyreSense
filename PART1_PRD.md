# TyreSense PART 1: data, baselines and the static GRU  (weeks 1-5)

Read AGENT_RULES.md first. Work on one task (T1.x) at a time.
Goal of Part 1: a clean dataset, honest baselines, a GRU that forecasts 20 laps ahead,
leakage tests, and an app page that replays a stint with a forecast. Demoable at week 5.

## Project in one paragraph
TyreSense forecasts how many laps an F1 tyre has left, from public FastF1 data only.
The final claim (tested in Part 2): does a forecast updated every lap beat one issued at
stint lap 2 and never updated? Part 1 builds everything that claim needs. Wear is inferred
from lap times only. A negative result later is valid.

## Data and cleaning
- Source: FastF1 only (laps, TyreLife, compound, weather, track status). Download once,
  save as parquet in data/ (gitignored). Pin the FastF1 version. Python 3.11.
- Splits by race: train 2022-2023, validation 2024, test 2025 (locked until the end).
  Check 2025 is complete in FastF1.
- Drop: wet-tyre races, lap 1, in-laps, out-laps, SC/VSC laps, deleted or inaccurate laps,
  missing fields, stints under 8 clean laps. One row per filter into reports/attrition.csv;
  rows must reconcile to the clean total.
- tyre_age = FastF1 TyreLife, never the row index. Gaps of up to 3 laps are kept and marked
  in gap_mask; a longer gap splits the stint.
- Fuel correction: lap_time_fc = lap_time - c * (total_laps - lap_number), c = 0.03 s/lap
  (assumed, say so in the report). Sensitivity later at 0.02 and 0.04.
- Usable-life label: first stint lap where lap_time_fc is 1.0 s slower than the stint's
  early-laps baseline. If never reached: censored = True, keep the stint.
  The threshold is a chosen definition, not a measurement.
- laps table columns: race_id, track_id, driver, stint_id, stint_lap, tyre_age, compound,
  lap_number, total_laps, lap_time_s, lap_time_fc_s, track_temp_c, track_status, gap_mask.
- stints table columns: stint_id, baseline_time_s, life_laps, censored.

## Inputs and models
Inputs (7): track, compound, tyre_age, lap fraction (fuel proxy), pace relative to the
driver's own reference, track temperature, track status.
Output: fuel-corrected lap-time residual for the next 20 laps (H = 20).

| Model | What it is |
| --- | --- |
| A baseline curve | Quadratic in tyre_age per track and compound, fuel-corrected, L2 penalty |
| B ridge | Linear model on the same 7 inputs |
| C XGBoost | Same inputs, strong static baseline |
| D GRU residual | GRU predicts what the baseline curve misses |
| RNN, LSTM | Same size as D, for EXP-7 only |

Baseline curve must be causal: before stint lap 3 use the track/compound curve; from lap 3
use the mean of the stint's observed first laps. Labels may use hindsight, inputs never.

GRU architecture: per-lap input 14 (7 numeric + track embedding 4 + compound embedding 3),
Dropout 0.1, GRU 14 to 32 (1 layer), Dropout 0.2, Linear 32 to 16, ReLU, Dropout 0.2,
Linear 16 to 20. About 5k parameters. Print the real count and check it.

Training defaults: target = standardised residual; compare MSE and Huber (delta 1.0);
Adam lr 1e-3, batch 64, weight decay 1e-4, gradient clip 1.0, ReduceLROnPlateau, early
stopping patience 15. Selection by race-grouped 3-fold CV inside 2022-2023. Seeds 42-44
(5 seeds for EXP-7). Runs save config + seed + metrics in runs/<name>/. Trains on CPU.

## Leakage tests (pytest)
1. No race_id appears in two splits.
2. Changing lap t+1 or later never changes the forecast at lap t.
3. Normalisers, baseline curves and embeddings are fitted on training races only; unseen
   tracks map to a shared "unk" embedding.
4. Final stint length and life_laps are never model inputs.
5. The baseline curve is causal (see above).

## Tasks (do in order; each is 1-2 hours)

| ID | Task | Done when |
| --- | --- | --- |
| T1.1 | Repo, pinned requirements.txt, config.yaml, download.py for 2022-2025 to parquet, .gitignore data/ | One race loads from cache with no network |
| T1.2 | clean.py: filters, attrition.csv, fuel function (student writes it), usable-life labels with censoring | Cleaning tests pass; synthetic fuel slope removed exactly; attrition reconciles |
| T1.3 | split.py by race + leakage tests 1, 3, 4 | Tests green |
| T1.4 | features.py: causal baseline curve, 7 inputs, sequences with gap mask; leakage tests 2, 5 | Tests green |
| T1.5 | baselines.py (A, B, C) + evaluate.py: MAE at horizons 1, 5, 10 and censored share per compound | Baseline table in reports/ |
| T1.6 | mlp_numpy.py (student writes it): 7-16-8-1 MLP, backprop, gradient check, SGD+momentum then Adam; EXP-1 | Max relative gradient error < 1e-6; loss within 5% of PyTorch |
| T1.7 | models.py (RNN, LSTM, GRU nn.Modules) + train.py with hand-written loop; pilot run, dated note recording pilot MAEs | Pilot numbers saved in runs/ |
| T1.8 | Run EXP-2, 3, 6, 7 (table below) | Results table per experiment in reports/ablation.csv |
| T1.9 | app.py: one Streamlit page, replay a stint from cache, lap slider + play/back/forward, chart of lap times with forecast line, remaining-life number (no band, no pit window yet) | First render < 3 s from cache; each step < 1 s; app sees only laps up to the current one |

## Experiments in Part 1
Protocol: same race splits, fixed seeds, hypothesis written before running. Expected
outcomes are hypotheses, not results.

| ID | Experiment | Metric | Hypothesis |
| --- | --- | --- | --- |
| EXP-1 | NumPy MLP vs PyTorch, gradient check, SGD+momentum then Adam | Max relative gradient error; loss gap | Error < 1e-6; loss within 5% |
| EXP-2 | ReLU, tanh, sigmoid; MSE vs Huber on MLP and GRU | MAE; 95th percentile error | ReLU trains faster; Huber helps on tails |
| EXP-3 | SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam; batch 16, 64, 256 | Epochs to target loss; final MAE | Adam or RMSProp fastest |
| EXP-6 | Curve, ridge, XGBoost, GRU (forecast issued once at lap 2) | MAE at horizons 1, 5, 10; parameter count | XGBoost at or near the best static |
| EXP-7 | Vanilla RNN vs LSTM vs GRU, same parameter count, 5 seeds | MAE at horizons 1 and 10 | Gated models at least as good |

## Part 1 is finished when
Leakage tests green, baseline table and EXP-1/2/3/6/7 results saved, the app replays a
stint with a forecast, and notes/explain.md has an entry for every task.

## What Part 2 expects from Part 1 (keep these names stable)
- clean.py: load_clean_laps(config) -> laps dataframe; make_stint_table(laps) -> stints
- features.py: build_sequences(laps, stints, fit_info) -> inputs, targets, gap_masks
- models.py: GRUForecaster (nn.Module, forward returns 20-lap residual forecast)
- train.py: train_model(config, train_data, val_data) -> saved model + runs/ folder
- evaluate.py: mae_at_horizons(pred, target, horizons)
- app.py: replay page that Part 2 extends
