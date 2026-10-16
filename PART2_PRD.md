# TyreSense Part 2 PRD (v2, replaces the old PART2_PRD.md)

Status date: 16 Oct 2026. Part 1 is complete (23/23 tests). This file lists every task left, in the order to do them.
Numbers marked "target" are proposals, not results. "verify" means check the source before citing.

## 0. How to use this file with Antigravity

Paste the block below at the start of every request, then paste ONE task (T2.x) at a time.

```
Read AGENT_RULES.md first and follow it. Work only on the task below.
Files <= 200 lines, functions <= 30 lines, plain functions, no lambdas, no new libraries.
Before coding: list your plan in 3-5 bullets and wait for my OK.
After coding: write the pytest, run it, show me the output, and give a plain-English
explanation (what it does, why, what could go wrong). Do not touch the 2025 test split.
Never invent numbers. If a result is unexpected, report it as it is.
```

After each task: read every line, add 3 lines to notes/explain.md (what, why, what breaks), commit.
Hand-write these yourself, with Antigravity only reviewing: the k-update loop (T2.9) and the PWE metric (T2.11). Teachers ask about these most and they are short.

## 1. Problems this version fixes (from the review of project_overview.md)

| # | Problem found | Where fixed |
| --- | --- | --- |
| 1 | Only 2 races (1,663 laps, 98 stints) are in data/; validation 2024 and test 2025 are not downloaded | T2.1 |
| 2 | Five proposed enhancements change the inputs from 7 to 8; Part 1 results would go stale | T2.2 to T2.5: flags, then one re-run |
| 3 | RNN 2,554, LSTM 7,162 and GRU 5,626 parameters are not a similar budget (EXP-7) | T2.6 |
| 4 | EXP-4 (regularisation) and EXP-5 (early stopping) are not in Part 2 | T2.7 |
| 5 | "Bayesian", "temperature scaling" and "guarantee" are wrong or overclaimed | Section 2 wording rules |
| 6 | test_leakage.py and test_live.py do not exist | T2.9 |
| 7 | App shows "Live Fuel on Board" as if measured | T2.12: label it "assumed" |

## 2. Wording rules for the report and the viva

- k is a "wear multiplier updated by a normalised error-correction rule". Not Bayesian, not a Kalman filter (Kalman is an optional extra).
- The interval fix is an "interval scale factor fitted on training folds, coverage checked on validation". Not temperature scaling, and nothing is guaranteed.
- "Frozen" only ever means FROZEN mode (forecast issued at stint lap 2). The 2025 evaluation uses "the final locked model".
- Usable life is a chosen definition (+1.0 s over the stint baseline), not a measured quantity.
- The Mercedes-AMG / Imperial 2025 and Montana State 2025 studies and the per-circuit fuel figures come from notes I could not verify. Open the papers before citing them.

## 3. Order of work and gates

```
T2.1 data -> T2.2 flags -> T2.3 clean -> T2.4 features -> T2.5 re-run Part 1
   GATE A: freeze data and features. No changes to inputs after this point.
-> T2.6 models -> T2.7 EXP-4/5 -> T2.8 EXP-8 -> T2.9 live + leakage tests
-> T2.10 uncertainty -> T2.11 EXP-9 -> T2.12 app
   GATE B: all tests green, EXP-1 to EXP-10 have numbers.
-> T2.13 test season (once) -> T2.14 report and viva
```

Suggested time (scale to your deadline, ask your teacher for the real date): week 1 = T2.1 to T2.3; week 2 = T2.4 to T2.6; week 3 = T2.7 to T2.9; week 4 = T2.10 and T2.11; week 5 = T2.12 and T2.13; week 6 = T2.14 and rehearsal.

Cut order if time runs short: Kalman extra, pace slider, second sensitivity table, seeds 5 to 3, EXP-4/5 depth. Never cut: T2.1, T2.5, T2.9, T2.10, T2.11, leakage tests.

## 4. Tasks

### T2.1 Full data download (src/download.py, make data)
- Download all dry-weather races 2022-2025 to data/raw/ (parquet). Resume if interrupted; skip races already cached; a second run makes no network call.
- Print a table: season, races found, races downloaded, races skipped (wet or failed).
- Before using 2025, confirm all rounds loaded. If 2025 is incomplete, say so and use the completed rounds only.
- Done when: test_download passes offline; the table is saved to reports/races.csv.
- Warning: each race takes minutes. Run it overnight, in batches by season.

### T2.2 Config flags (config.yaml, src/config.py)
Add these settings, all defaulting to the Part 1 behaviour so nothing changes until you switch them on:

```
fuel:      mode: constant | track_scaled     base_c: 0.03    ref_km: 5.0
spike:     enabled: false   threshold_s: 2.0   recover_s: 1.0   window: 3
features:  temp_delta: false
model:     head: free | monotone    hidden: {gru: 32, lstm: 27, rnn: 60}
loss:      name: huber | asym_huber   delta: 1.0   positive_weight: 0.5
live:      eta: 0.3   k_min: 0.5   k_max: 2.0   min_g: 0.05   smooth_laps: 3
mc:        passes: 30   sigma_k: 0.1
```

Also add a circuit_length_km table for every circuit that appears in the data (verify each length on an official circuit page, and note the source in a comment).
Done when: the config loads and the old tests still pass.

### T2.3 Cleaning upgrades (src/fuel.py, src/clean.py)
1. Fuel, keep your correct sign: `lap_time_fc = lap_time - c * (total_laps - lap_number)`. In track_scaled mode `c = base_c * circuit_length_km / ref_km`. An unknown circuit falls back to base_c and is listed in a warning.
2. Spike filter, two functions:
   - `flag_spikes_offline(stint)`: a lap is a spike if its fuel-corrected time is more than threshold_s above the median of the previous 3 clean laps AND the next lap is within recover_s of that median. Used only for building training data.
   - `flag_spikes_causal(history)`: same, but without the "next lap" condition. Used by the replay, the app and live correction.
3. Flagged laps go into gap_mask. Count them as one extra row in attrition.csv.
4. tyre_age must be the true FastF1 tyre life, not the row index. Gaps up to 3 laps are kept; a longer gap splits the stint.
- Tests: synthetic stint where the fuel slope is removed exactly in both modes; a synthetic spike is flagged offline and causally; a real slow lap with no recovery is not flagged offline.

### T2.4 Features (src/features.py, src/mlp_numpy.py)
- Add input 8: `track_temp_delta_c = temperature now - temperature at stint start` (known at forecast time, so causal).
- MLP becomes 8-16-8-1. Re-run the gradient check (target error below 1e-6) and PyTorch parity.
- Test: the delta at stint lap 1 is exactly 0; perturbing a later lap's temperature does not change earlier inputs.

### T2.5 Re-run Part 1 on full data (src/ablation.py)
- Switch on: track_scaled fuel, spike filter, temp_delta. Rebuild laps.parquet, stints.parquet, attrition.csv.
- Re-run baselines A, B, C, the pilot GRU, and EXP-2, 3, 6, 7. Keep the old Part 1 tables in reports/part1_old/ so you can compare.
- Add the sensitivity table: constant 0.03 vs track_scaled vs base 0.02 and 0.04 (run on ridge and XGBoost only, to save time).
- Write a dated note in notes/explain.md with the new clean-lap count per season. If the clean data is still small (under about 5,000 laps in training), say so and keep the model at hidden 32.
- **GATE A: freeze data and features here.**

### T2.6 Models (src/models.py, src/train.py)
1. Matched sizes for EXP-7. The three models use the same inputs and head, and hidden sizes chosen so the total parameter counts are within 5% of each other. Starting guess: GRU 32, LSTM 27, RNN 60 (re-check after the input count became 15; adjust and keep the numbers in config). Add `test_models` assertions for the 5% rule.
2. Monotone head (option): `steps = linear(...)` then `degradation = g_now + torch.cumsum(torch.relu(steps), dim=1)`. g_now is the mean of the last 3 fuel-corrected laps minus the stint baseline. With this head the target is future degradation relative to the stint baseline; write this in model.yaml.
3. Asymmetric Huber loss: Huber, but laps slower than predicted (actual above prediction) get weight positive_weight (0.5), others 1.0. Default stays Huber unless EXP-2 shows otherwise.
- Tests: with the monotone head, every forecast is non-decreasing; loss weights are applied to the right sign.

### T2.7 EXP-4 and EXP-5 (src/ablation.py)
- EXP-4: L2 in {0, 1e-5, 1e-4, 1e-3, 1e-2}; dropout in {0, 0.1, 0.2, 0.3, 0.5}; L1 (Lasso) on the linear model, count non-zero weights. Metric: validation MAE, train-validation gap.
- EXP-5: early stopping with patience 5, 15, 30 vs fixed 150 epochs. Metric: validation MAE, epochs used.
- Hypothesis written first in notes/explain.md. Report negative results as they are.

### T2.8 EXP-8 gradient flow (src/gradflow.py)
- For vanilla RNN, LSTM, GRU, compute the norm of dL/dh at each time step for sequence lengths 10, 25, 50, 100. Real stints are at most about 40 laps, so for 50 and 100 tile real stints end to end and state that this is a stress test. Report the first-step to last-step norm ratio.
- Gradient clipping at 0.5, 1, 5: report whether training diverges.
- Plot to reports/figures/gradflow.png. Hypothesis: vanilla RNN decays most; on real-length stints the gap is small. Report what you see.

### T2.9 Live correction, inference modes and leakage tests (src/live.py, src/evaluate.py)
Hand-write the core loop; Antigravity reviews. Reference:

```
k = 1.0; k_history = []
for each stint lap t >= 3:
    pred = base_t + k * g_t
    err  = actual_t - pred
    if lap t is not flagged as a spike (causal flag) and abs(g_t) > min_g:
        k_raw = k + eta * g_t * err / (g_t**2 + 0.05)
        k_history.append(k_raw)
        k = clip(mean(k_history[-3:]), k_min, k_max)
    forecast the next 20 laps with k
```

- FROZEN: forecast issued at stint lap 2, never recomputed. ROLLING: recomputed every lap with the new laps. E = ROLLING plus the k above. Remaining life = first horizon where forecast degradation reaches the cliff threshold; if none within 20 laps, report "more than 20" (censored).
- tests/test_live.py: on a synthetic stint with a known wear rate and added noise (sigma about 0.3 s), k moves toward the true rate within 5 laps and stays inside [0.5, 2.0]; k is unchanged before lap 3 and on a flagged lap.
- tests/test_leakage.py: (1) no race in two splits; (2) changing lap t+1 or later does not change the forecast or the spike flag at lap t; (3) scalers, baseline curves and track vocabulary are fitted on training races only; (4) final stint length and life_laps are never inputs; (5) the app and replay only receive laps up to t.

### T2.10 Uncertainty (src/uncertainty.py), EXP-10
- MC-dropout: keep only the Dropout layers in train mode at inference, 30 passes. In E also draw `k_sample = clip(k + Normal(0, sigma_k), k_min, k_max)` for each pass. Each pass gives a remaining life; take p10, p50, p90.
- Interval scale factor: `q_scaled = p50 + s * (q - p50)`. Fit one s by grid search on out-of-fold training predictions (3-fold CV, folds grouped by race) so the 80% band covers about 80% there.
- Report on validation: coverage at 50, 60, 70, 80, 90% before and after scaling, mean width, and a reliability plot. Target: 80% band covers 75-85% after scaling. If it does not, say so.

### T2.11 EXP-9 core claim (src/evaluate.py)
Compare at stint laps 3, 5 and 7: XGBoost (static, for reference), D-FROZEN, D-ROLLING, E.
- Headline metric, **pit window error (PWE)** = absolute gap in laps between the predicted (p50) and the actual usable-life lap, on uncensored stints. Also report the share of stints within 2 laps.
- Censored stints: report the share where the model correctly says "more than 20", and the censored share per compound.
- 5 seeds; paired bootstrap over races (1,000 resamples) for each difference; report mean, standard deviation and the confidence interval.
- Answer two questions in the report: (1) does ROLLING beat FROZEN? (2) does the k correction add anything over D-ROLLING? Either answer is a valid result. Possible reasons for a negative result: lap times alone carry too little signal, or the GRU already adapts by itself.
- Sensitivity: cliff threshold 0.5, 0.75, 1.0, 1.25, 1.5 s (rerun labels and the final model only).

### T2.12 App upgrade (app.py)
- Add: 80% band, grey FROZEN ghost line, remaining life as a range ("8-10 laps"), pit window `[t + p10 - 2, t + p50]` (shown only when p10 and p90 exist, never below lap 3), k display ("1.18, faster wear than usual"), model selector A, D, E.
- Relabel the fuel card "Assumed fuel on board" with a one-line caption.
- Keep the speed targets: first load under 3 s, each step under 1 s (test_app).
- Pace slider is optional and must carry the caption "model association, not a causal effect".

### T2.13 Test season, opened once (src/final_eval.py)
- Run only when GATE B is passed and the configuration is frozen. Evaluate the final locked model on 2025 once. Save to runs/final/ with the commit hash. Never re-run to improve it.
- Error analysis: the 10 worst stints, one line each (cause: traffic, rain, strategy, data problem, model).

### T2.14 Report and viva
- Report sections: problem, data and attrition table, methods, 10 experiments with the syllabus map, results (including negatives), limits, ethics and credit, AI-assistance statement if your course requires it (check NMIMS policy).
- Update the viva file with the cards for fuel scaling, the spike rule, the monotone head, the asymmetric loss, PWE, the interval scale factor and k. Rehearse the 6-minute demo twice.

## 5. Experiment checklist

| ID | Unit | Status | Task |
| --- | --- | --- | --- |
| EXP-1 | 2 | done (re-check at 8 inputs) | T2.4 |
| EXP-2 | 2 | re-run, plus asym loss and monotone head | T2.5, T2.6 |
| EXP-3 | 3 | re-run | T2.5 |
| EXP-4 | 3 | to do | T2.7 |
| EXP-5 | 3 | to do | T2.7 |
| EXP-6 | 1 | re-run | T2.5 |
| EXP-7 | 5 | re-run with matched sizes | T2.6 |
| EXP-8 | 5 | to do | T2.8 |
| EXP-9 | 2, 5 | to do | T2.11 |
| EXP-10 | 3 | to do | T2.10 |

Unit 4 (convolutional networks) is not covered by this project. Confirm with your teacher whether it is needed; if so, add a separate small CNN or ResNet-18 exercise and label it as outside TyreSense.

## 6. Definition of done

- [ ] pytest green, including test_leakage and test_live
- [ ] attrition.csv reconciles to the clean-lap total for every season
- [ ] reports/ablation.csv has EXP-1 to EXP-10 with 3 seeds (5 for EXP-7 and EXP-9)
- [ ] 2025 evaluated exactly once, result saved in runs/final/
- [ ] app demo under 7 minutes, three commands in the README
- [ ] notes/explain.md has an entry for every task, and you can explain each one
- [ ] no raw or cached F1 data in the repository

## 7. Risks

| Risk | What to do |
| --- | --- |
| Clean data is still small after the full download | Keep the model small, report the limit, lean on the baseline plus correction |
| FastF1 download fails or is slow | Resume from cache; if a season is incomplete, use the rounds you have and say so |
| Enhancements do not help | Report them as ablations; keep the flag off if validation does not improve |
| EXP-9 is negative | Report it plainly; this is still a complete project |
| Time runs short | Follow the cut order in section 3 |
