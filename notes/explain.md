# TyreSense Explanation Notes

## Task T1.1
- What it does: Sets up the repository configuration, pinned dependencies, FastF1 caching, and parquet download pipeline.
- Why: Ensures reproducible offline access to race session data without hitting live F1 APIs repeatedly.
- What breaks: If FastF1 API formats change or network fails during initial download, session extraction fails before caching.

## Task T1.2
- What it does: Applies sequential filters, fuel correction, gap-aware stint splitting, and 1.0 s degradation threshold labelling with censoring.
- Why: Isolates genuine tyre degradation from fuel burning and disruption laps (SC/VSC, pit entries/exits) while tracking all dropped data.
- What breaks: Stints with high lap noise or early traffic can artificially cross the 1.0 s threshold or get split inappropriately.

## Task T1.3
- What it does: Splits data strictly by race season (2022-2023 train, 2024 val, 2025 test) and guarantees train-only fitting with unseen track fallback.
- Why: Prevents data leakage across races and guarantees models never see future season tracks or target labels in input features.
- What breaks: If a race ID string format changes or test season data is prematurely exposed, evaluation validity is compromised.

## Task T1.4
- What it does: Fits quadratic baseline curves per track/compound with causal stint shifting, extracts 7 causal features, and builds 20-lap horizon residual targets.
- Why: Supplies causal inputs and an initial degradation reference so deep learning models only have to learn higher-order residual patterns.
- What breaks: Changing future laps could leak hindsight if relative pace or curve offsets used unobserved future laps instead of strictly causal lap indexing.

## Task T1.5
- What it does: Implements baseline Models A (causal quadratic curve), B (Ridge linear regression), and C (XGBoost), computing MAE at horizons 1, 5, and 10 with compound-level censored share reporting.
- Why: Establishes static benchmarks that sequential recurrent models must demonstrably outperform in later experiments.
- What breaks: Evaluation on stints with extreme outliers or unrepresented compound classes can distort multi-horizon error metrics.

## Task T1.6
- What it does: Implements a 7-16-8-1 MLP in pure NumPy with backpropagation, analytical vs numerical gradient check, and SGD+momentum and Adam optimizers.
- Why: Demonstrates fundamental deep learning mechanics and loss equivalence with PyTorch for viva examination defense.
- What breaks: Finite difference precision limits or wrong activation derivatives will cause gradient checking relative error to exceed 1e-6.

## Task T1.7
- What it does: Defines PyTorch RNN, LSTM, and GRU forecasters (~5.6k parameters) and runs a hand-written training loop on CPU using Huber loss.
- Why: Implements model D to learn sequential residual patterns over the causal baseline curve.
- What breaks: Unmasked loss over NaN horizon entries when stints terminate before horizon 20 will cause loss gradient explosion.
- Pilot run note (Date: 2026-10-15): GRU trained for 3 epochs on CPU (5,626 parameters). Validation MAE h=1: 1.079 s, h=5: 1.202 s, h=10: 1.185 s. Recorded in runs/pilot/metrics.json.

## Task T1.8
- What it does: Executes ablation experiments EXP-2 (activations/losses), EXP-3 (optimizers/batch sizes), EXP-6 (lap 2 static vs dynamic forecasts), and EXP-7 (RNN/LSTM/GRU across 5 seeds).
- Why: Validates architectural design choices systematically before proceeding to online dynamic updates.
- What breaks: Inconsistent seed initialization or evaluation on differing race splits would invalidate ablation comparison integrity.

## Task T1.9
- What it does: Implements a single-page Streamlit application that replays cached stints lap-by-lap with causal 20-lap forward forecasts and remaining tyre life estimation.
- Why: Provides an interactive inspection tool to visually verify causal tyre degradation tracking and cliff onset without hindsight leakage.
- What breaks: Inspecting unobserved future stint laps or attempting online network downloads would violate causality and latency constraints.

## Task T2.1
- What it does: Implements multi-season FastF1 downloading with automatic disk cache checking and logs tracking data to reports/races.csv.
- Why: Provides resilient batch downloading across 2022-2025 seasons while guaranteeing zero network overhead on repeated runs.
- What breaks: Live F1 API rate limits or network drops during initial downloads could produce incomplete parquet chunks without resumption checks.

## Task T2.2
- What it does: Adds backward-compatible configuration flags and an official FIA circuit lengths lookup table to config.yaml.
- Why: Allows enabling advanced physics (track-scaled fuel burn, spike filters) without breaking existing Part 1 models or tests.
- What breaks: Missing circuit names or mismatched keys in the circuit lookup table could cause fuel scaling calculations to fail silently.

## Task T2.3
- What it does: Upgrades fuel correction with track-distance scaling and implements offline (with recovery check) and causal anomaly spike filters.
- Why: Eliminates residual slope distortion on short/long tracks and prevents momentary traffic or driver mistakes from corrupting tyre degradation curves.
- What breaks: Setting spike thresholds too low could filter genuine sharp degradation cliffs, while failing to check recovery would mask permanent pace drops.

## Task T2.4
- What it does: Adds input 8 (track_temp_delta_c) to feature building and updates the NumPy MLP to an 8-16-8-1 architecture with verified analytical gradients.
- Why: Captures track cooling/heating trends causally across stints while verifying mathematical rigor and PyTorch parity on the expanded input dimension.
- What breaks: Calculating temperature delta using future stint laps would leak lookahead information into early lap features.

## Task T2.5
- What it does: Re-runs Part 1 cleaning and experiments with track-scaled fuel, spike filtering, and temperature delta; generates the fuel sensitivity comparison table.
- Why: Empirically validates that track-scaled fuel improves horizon MAE over static 0.03 s/lap, establishing frozen data and features at Gate A.
- What breaks: Adding features or modifying cleaning filters after Gate A would invalidate all downstream sequence modeling and uncertainty calibrations.
- Dated note (Date: 2026-10-18): Clean data count: 2023 season has 1,663 clean laps across 98 stints (877 Bahrain, 786 Saudi Arabia). Training data is currently under 5,000 laps, so model hidden dimension is kept at 32 to prevent overfitting. Gate A is passed; data and features are frozen.

## Task T2.6
- What it does: Balances parameter counts across GRU (32), LSTM (27), and RNN (58) within 1.6% (< 5%), implements an optional monotone non-decreasing head, and introduces asymmetric Huber loss.
- Why: Guarantees fair architectural benchmarking for EXP-7, enforces physical monotonicity where tyres cannot spontaneously regain grip, and down-weights one-off driver mistakes.
- What breaks: An unbalanced parameter budget would confound architectural comparisons with raw capacity differences, while decreasing degradation outputs violate physical realism.
