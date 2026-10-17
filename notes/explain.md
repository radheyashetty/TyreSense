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

## Task T2.7 (Hypothesis)
- EXP-4 Hypothesis: Regularization follows a U-shaped validation curve where moderate weight decay (1e-4) and dropout (0.2) minimize validation MAE without underfitting. L1 Lasso penalty produces sparse feature weights on the linear baseline.
- EXP-5 Hypothesis: Early stopping with patience 15 converges to within 2% of the validation MAE of fixed 150 epochs while stopping early and saving significant training iterations.

## Task T2.7
- What it does: Executes EXP-4 (L2 decay sweep, dropout rate sweep, and Lasso sparsity) and EXP-5 (early stopping patience vs fixed epochs), recording results to reports/ablation.csv.
- Why: Validates regularization hyperparameters against overfitting and determines optimal training stopping criteria to conserve compute.
- What breaks: Over-regularizing with high L1 penalty zeroes out essential tyre wear features, causing validation MAE to spike significantly.

## Task T2.8
- What it does: Computes step-wise gradient flow norms ||dL/dh_t|| for RNN, LSTM, and GRU across sequence lengths 10, 25, 50, 100 and clipping thresholds, saving the plot to reports/figures/gradflow.png.
- Why: Empirically demonstrates the vanishing gradient problem over multi-lap horizons and verifies that gradient clipping prevents training divergence.
- What breaks: Very long unrolled sequences without gating or residual connections suffer exponential gradient decay, completely freezing weights for early stint laps.

## Task T2.9
- What it does: Implements online adaptive tyre wear correction (k-multiplier update) in src/live.py, verifies k-factor dynamics in tests/test_live.py, and formalizes all 5 data leakage tests in tests/test_leakage.py.
- Why: Provides Model E with online adaptation to real-time track abrasion while proving mathematically and empirically that future laps never leak into past features.
- What breaks: Allowing k updates on traffic spike laps or before lap 3 causes lap noise to blow up the multiplier outside [0.5, 2.0].

## Task T2.10
- What it does: Implements 30-pass Monte Carlo Dropout inference in src/uncertainty.py, extracts P10, P50, P90 usable life intervals, and fits interval scale factor s to achieve 80% coverage on validation stints.
- Why: Provides race engineers with well-calibrated confidence intervals rather than over-confident point forecasts for pit stop strategy.
- What breaks: Raw uncalibrated MC-dropout intervals are typically too narrow and significantly under-cover true stint variance without post-hoc scaling.

## Task T2.11 (Hypothesis)
- EXP-9 Hypothesis: Online adaptive Model E and rolling residual forecasting significantly reduce Pit Window Error (PWE) compared to frozen lap-2 baselines, with narrower paired bootstrap confidence intervals.

## Task T2.11
- What it does: Implements Pit Window Error (PWE) evaluation comparing FROZEN, ROLLING, and Model E across stint laps 3, 5, 7, paired bootstrap over races, and cliff threshold sensitivity sweeps in src/claim_eval.py.
- Why: Rigorously verifies the core project claim that online telemetry adaptation produces superior pit stop timing recommendations compared to static pre-race degradation curves.
- What breaks: Relying on fixed early-stint forecasts fails to capture intra-race track temperature shifts and tyre graining, leading to missed pit windows by multiple laps.

## Task T2.12
- What it does: Upgrades app.py with a shaded 80% MC-Dropout uncertainty envelope, grey FROZEN lap-2 ghost line comparison, strategic pit window card [P10-2, P50], real-time wear multiplier k gauge, Model picker (A, D, E), and relabels fuel to "Assumed fuel on board".
- Why: Provides race engineers with an intuitive visual decision-support interface that captures predictive uncertainty and demonstrates the real-time value of telemetry feedback.
- What breaks: Running dense Monte Carlo simulations directly inside the per-step interactive render loop exceeds the 1.0s latency budget; parametric uncertainty bounds calibrated via EXP-10 preserve real-time responsiveness.

## Task T2.13
- What it does: Evaluates the locked Model E on held-out test data exactly once, storing test horizon MAEs (h=1: 1.091s, h=5: 1.209s, h=10: 1.322s) and git commit hash to runs/final/metrics.json, and records diagnostic root causes for worst-case stints in reports/error_analysis.csv.
- Why: Provides an uncompromised, un-retuned final test benchmark proving true generalizability without data leakage or test set overfitting.
- What breaks: Opening and repeatedly running training or hyperparameter sweeps against the held-out test split causes subtle test leakage and optimistic performance bias.

## Task T2.14
- What it does: Compiles the final comprehensive technical project report in reports/final_report.md covering all 10 experiments, methods, ethical implications, and final test results, and finalizes viva preparation cards.
- Why: Provides complete scientific and engineering documentation for course evaluation, defense, and reproducibility.
- What breaks: Omitting negative findings (such as RNN gradient collapse or L1 over-sparsification) weakens the scientific validity of the research conclusions.

---

## Viva Preparation Cards (Key Concepts & Defense)

### 1. Fuel Scaling
- **Concept**: $c_{\text{track}} = c_{\text{base}} \times (\text{Circuit Length} / 5.0\text{ km})$.
- **Why**: Fuel mass penalty per lap depends directly on lap distance and engine duty cycle. A 7 km circuit (Spa) burns more fuel per lap than a 3.3 km track (Monaco), so a uniform $0.03\text{ s/lap}$ constant over-penalizes short tracks and under-penalizes long tracks.
- **Empirical Proof**: Track-scaled fuel yields the lowest residual pace MAE ($0.765\text{ s}$ vs $0.768\text{ s}$ for constant).

### 2. Causal Spike Filtering
- **Concept**: A lap is flagged as a traffic/anomaly spike if $t_{\text{lap}} - t_{\text{lap}-1} > 2.0\text{ s}$ and subsequent laps recover ($t_{\text{lap}+1} - t_{\text{lap}} < -1.0\text{ s}$).
- **Why**: In causal online inference, future laps are unseen. In `flag_spikes_causal`, a lap is flagged immediately if pace degrades $> 2.0\text{ s}$ relative to rolling median pace, preventing dirty air or momentary yellow flags from distorting the degradation curve.
- **What Breaks**: If spikes aren't flagged, single-lap traffic anomalies corrupt the wear multiplier $k$.

### 3. Monotone Head
- **Concept**: $\hat{y}_{\text{res}}[h] = \hat{y}_{\text{res}}[0] + \sum_{i=1}^h \exp(w_i) \cdot \Delta$.
- **Why**: Physical tyre wear cannot heal during a dry stint. Enforcing positive incremental outputs via exponential parametrization guarantees non-decreasing degradation predictions.
- **What Breaks**: Standard unconstrained dense heads frequently predict unphysical pace recovery at long horizons ($h \ge 10$).

### 4. Asymmetric Huber Loss
- **Concept**: $\mathcal{L}_{\text{asym}}(r) = w_+ \cdot \mathcal{L}_{\text{Huber}}(r)$ if $r > 0$ else $(1 - w_+) \cdot \mathcal{L}_{\text{Huber}}(r)$.
- **Why**: Underestimating tyre degradation ($r > 0$, where actual lap times are slower than predicted) causes cars to be caught out on worn tyres past the cliff, losing seconds per lap. Asymmetric weighting penalizes optimistic errors more severely.

### 5. Pit Window Error (PWE)
- **Concept**: $\text{PWE} = |\hat{L}_{\text{cliff}} - L_{\text{cliff}}|$, measuring error in laps until the $+1.0\text{ s}$ degradation cliff.
- **Why**: MAE in seconds does not directly reflect strategic utility. Race strategists need to know which lap to pit; PWE measures strategy error in actionable units (laps).

### 6. Uncertainty Interval Scale Factor ($s$)
- **Concept**: $Q_{10} = P_{50} + s \cdot (P_{10} - P_{50})$, $Q_{90} = P_{50} + s \cdot (P_{90} - P_{50})$.
- **Why**: Raw MC-dropout intervals are typically under-dispersed due to correlated epistemic sampling. Calibrating $s$ via grid search matches empirical coverage to the nominal 80% credible interval on validation data.

### 7. Online Wear Multiplier ($k$)
- **Concept**: $k_{\text{new}} = \text{clip}\left(k + \eta \cdot \frac{g_t \cdot e_t}{g_t^2 + 0.05},\, 0.5,\, 2.0\right)$.
- **Why**: Adjusts the pre-race degradation slope $g_t$ up or down based on actual observed residual error $e_t$. The normalized denominator prevents division by zero when baseline degradation is small, and clipping bounds adaptation to physically realistic wear rates.
