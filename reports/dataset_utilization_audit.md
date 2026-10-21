# TyreSense — Maximum Eligible Dataset Utilization Audit Report

**Date of Audit**: 2026-10-22  
**Auditor**: Senior Python Software Engineer & ML Data Auditor  
**Audit Scope**: Permitted Training (2022–2023) and Validation (2024) Datasets Only  
**Final Verdict**: **EFFECTIVE USE VERIFIED WITH INTENTIONAL LIMITATIONS**  

---

## 1. Scope & Data-Access Isolation Declaration

This audit strictly evaluated the empirical utilization of the permitted Formula 1 telemetry datasets across the TyreSense pipeline.

- **Permitted Data Sources Audited**:
  - `data/processed/train_laps.parquet`: 2022–2023 Grand Prix races (38 races).
  - `data/processed/val_laps.parquet`: 2024 Grand Prix races (21 races).
  - `data/processed/stints.parquet`: Stint-level metadata, baseline pace, and censoring flags for permitted seasons.
  - `data/processed/laps.parquet`: Consolidated processed telemetry (permitted rows 2022–2024).
  - `reports/attrition.csv`: Pipeline cleaning attrition log.
- **Protected Data Strictly Excluded**:
  - `data/processed/test_laps.parquet` (**Locked 2025 Test Split**).
  - Raw Parquet race files for season 2025 (`data/raw/race_2025_*.parquet`).
  - **Confirmation**: Under Rule R3, zero reads, queries, samplings, or evaluations were performed on 2025 data.

---

## 2. Actual Sequence-Generation Implementation Trace

The sequence-generation and windowing mechanics were traced across [`src/features.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/features.py), [`src/train.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py), [`src/models.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/models.py), and [`src/eval_four_models.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/eval_four_models.py):

| Implementation Dimension | Real Code Implementation | Relevant Function & File |
| :--- | :--- | :--- |
| **Input Sequence Length ($L$)** | Fixed length $L = 10$ laps. | `extract_padded_windows` ([`src/train.py#L30`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L30)). |
| **History Conditioning** | Fixed-length sliding window with left zero-padding when $t < 10$. | `np.pad(win_num, ((pad_len, 0), (0, 0)), mode="constant")` ([`src/train.py#L41`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L41)). |
| **Example Generation Frequency** | **One example per clean lap** ($t = 1 \dots n$ for each stint). | `for t in range(1, num_laps + 1):` ([`src/train.py#L34`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L34)). |
| **Stint Independence** | Sliced strictly within each `stint_id`. | `for stint_id in laps_df["stint_id"].unique():` ([`src/train.py#L75`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L75)). |
| **Boundary Crossing** | **No**. Windows never cross stints, drivers, races, or season splits. | Sliced over `sub = laps_df[laps_df["stint_id"] == stint_id]` ([`src/train.py#L76`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L76)). |
| **Earliest Lap Handling** | Lap 1 produces a valid example (9 left-padded zero vectors + 1 lap vector). | `pad_len = seq_len - (t - start_idx)` ([`src/train.py#L36`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L36)). |
| **Window Overlap** | Consecutive windows within a stint overlap by up to 9 laps. | Windows advance by step $\Delta t = 1$ lap. |
| **Target Quantity** | Vector of 20 future horizon residuals: $r_{t+h} = y_{t+h} - B_{t+h}$. | `build_horizon_targets` ([`src/features.py#L101`](file:///C:/Users/radhe/Desktop/deep%20learning/src/features.py#L101)). |
| **End-of-Stint Truncation** | Targets extending past stint end are filled with `NaN` and masked out. | `targets[i, h - 1] = np.nan` ([`src/features.py#L105`](file:///C:/Users/radhe/Desktop/deep%20learning/src/features.py#L105)). |
| **Incomplete Target Handling** | Retained with boolean target mask: `mask = ~torch.isnan(targets)`. | `StintSequenceDataset` ([`src/train.py#L20`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L20)). |
| **Loss Masking** | Only non-NaN target slots contribute to Huber loss: `(loss * mask).sum() / mask.sum()`. | `masked_loss` ([`src/train.py#L104`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py#L104)). |

---

## 3. Exact Sequence Accounting

The exact counts were independently computed from [`data/processed/train_laps.parquet`](file:///C:/Users/radhe/Desktop/deep%20learning/data/processed/train_laps.parquet) and [`data/processed/val_laps.parquet`](file:///C:/Users/radhe/Desktop/deep%20learning/data/processed/val_laps.parquet):

| Sequence Metric | Training Split (2022–2023) | Validation Split (2024) | Verification Method |
| :--- | :---: | :---: | :--- |
| **Available Clean Laps** | 33,269 | 20,028 | Row count of processed split parquets. |
| **Distinct Stints** | 1,756 | 903 | `stint_id.nunique()` on clean parquets. |
| **Stints Excluded by Length** | 0 (0.0%) | 0 (0.0%) | Min stint length is 8 laps (all clean stints $\ge 3$ laps). |
| **Actual Generated Examples** | **33,269** | **20,028** | Length of PyTorch `StintSequenceDataset`. |
| **Examples per Stint (Min, Med, Mean, Max)** | 8, 17.0, 18.95, 53 | 8, 20.0, 22.18, 76 | Stint length distribution (1 window per clean lap). |
| **Examples with Full History ($t \ge 10$)** | 17,537 (52.71%) | 11,932 (59.58%) | Windows with 10 real observed laps. |
| **Examples with Padded History ($t < 10$)** | 15,732 (47.29%) | 8,096 (40.42%) | Laps 1–9 of each stint (left zero-padded). |
| **Duplicate Input Tensors** | 2 / 33,269 (0.006%) | 1 / 20,028 (0.005%) | Duplicate input tensors from identical early-lap telemetry. |
| **Total Input Tensor Slots ($N \times 10$)** | 332,690 | 200,280 | Total tensor capacity ($10 \text{ steps} \times N \text{ examples}$). |
| **Non-Padded Real Input Steps** | 253,742 (76.27%) | 159,676 (79.73%) | Sum of $\min(t, 10)$ across all stint laps. |
| **Zero-Padded Left Steps** | 78,948 (23.73%) | 40,604 (20.27%) | Padding steps for laps $t < 10$. |
| **Mean Lap Appearances as Input** | 7.63 times | 7.97 times | Non-padded steps divided by clean laps. |
| **Laps Never Used as Input** | **0** (0.0%) | **0** (0.0%) | Every clean lap serves as a window terminal step. |
| **Laps Only in Zero-Target Examples** | 1,756 (5.28%) | 903 (4.51%) | Final lap of each stint has 0 future targets inside stint. |

---

## 4. Target Masking & Effective Learning Signal

While 100% of clean laps act as sequence endpoints, **not all future target slots contain valid data** due to natural stint termination:

| Target Metric | Training Split (2022–2023) | Validation Split (2024) | Explanation |
| :--- | :---: | :---: | :--- |
| **Total Target Slots ($N \times 20$)** | 665,380 | 400,560 | 20 forward horizon slots per generated window. |
| **Valid (Non-NaN) Target Slots** | **328,350 (49.35%)** | **222,934 (55.66%)** | Real future lap residuals within the stint. |
| **Masked (NaN) Target Slots** | **337,030 (50.65%)** | **177,626 (44.34%)** | Laps beyond the physical end of the stint (pit stops). |
| **Examples with 0 Valid Targets** | 1,756 (5.28%) | 903 (4.51%) | Exactly 1 per stint (the final lap $t = n$ has no future laps). |
| **Valid Targets per Window (Mean)** | **9.87 laps** | **11.13 laps** | Validation mean is higher because 2024 stints are longer. |
| **Valid Targets per Window (Median)** | 9.0 laps | 11.0 laps | Median non-NaN future horizon length. |
| **Loss Normalization Denominator** | Batch `mask.sum()` | Batch `mask.sum()` | Mean loss is normalized strictly by valid target count. |

### Valid Target Counts by Forecast Horizon ($h = 1 \dots 20$)

$$\text{Valid Share}(h) = \frac{\text{Stints with Remaining Length} \ge h}{N_{\text{examples}}}$$

| Horizon ($h$) | Train Valid Count | Train Valid Share | Val Valid Count | Val Valid Share | Loss Contribution |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$h = 1$** | 31,513 | 94.7% | 19,125 | 95.5% | Very High (all laps except stint end) |
| **$h = 2$** | 29,757 | 89.4% | 18,222 | 91.0% | High |
| **$h = 3$** | 28,001 | 84.2% | 17,319 | 86.5% | High |
| **$h = 5$** | 24,489 | 73.6% | 15,513 | 77.5% | Moderate |
| **$h = 7$** | 20,977 | 63.1% | 13,707 | 68.4% | Moderate |
| **$h = 10$** | 15,935 | 47.9% | 11,096 | 55.4% | Reduced (half of stints have pitted) |
| **$h = 15$** | 9,288 | 27.9% | 7,390 | 36.9% | Low |
| **$h = 20$** | 5,171 | 15.5% | 4,571 | 22.8% | Sparse (only long stints remain) |

---

## 5. Training Consumption Verification

The execution path of the PyTorch `DataLoader` was traced from [`src/train.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/train.py):

- **Dataset Length**: 33,269 examples.
- **Batch Size**: 64 (default in `setup_pilot_loaders`).
- **`drop_last` Setting**: `False` (all examples are consumed).
- **Batches per Epoch**: $\lceil 33,269 / 64 \rceil = \mathbf{520\text{ batches}}$ (519 batches of 64, 1 batch of 53).
- **Examples Consumed per Epoch**: **33,269 examples (100.0%)**.
- **Epochs Recorded**:
  - `runs/gru_main/config.json`: `max_epochs = 25`, `patience = 5`, `lr = 0.001`, `seeds = [42, 123, 456]`.
  - Over a full 25-epoch run, an un-early-stopped model receives $25 \times 33,269 = \mathbf{831,725\text{ window presentations}}$.
- **Loss Execution**: `masked_loss` computes Huber loss across all batch elements and divides by the non-zero mask count. Windows with 0 valid targets contribute 0 to the numerator and 0 to the denominator, effectively producing zero gradient update without causing numerical instability.

---

## 6. Validation Coverage & Metric Reconciliation

In [`src/eval_four_models.py`](file:///C:/Users/radhe/Desktop/deep%20learning/src/eval_four_models.py), validation is evaluated at **cutoff lap 5** (`cutoff = 5`), producing **exactly one 20-lap forecast per stint**, not 20,028 sliding windows.

### Reconciliation of Reported Benchmark Figures

| Benchmark Metric | Evaluated Stints | Target Lap | Evaluated Sample Share | Underlying Mathematical Reason |
| :--- | :---: | :---: | :---: | :--- |
| **MAE at $h=1$** | **903 / 903** | Lap 6 | **100.0%** | Minimum stint length in 2024 is 8 laps; lap 6 exists in all 903 stints. |
| **MAE at $h=5$** | **836 / 903** | Lap 10 | **92.6%** | 67 stints pitted before lap 10. |
| **MAE at $h=10$** | **684 / 903** | Lap 15 | **75.7%** | 219 stints pitted before lap 15. |
| **Cliff PWE** | **362 / 903** | Variable | **40.1%** | Only uncensored stints have an observed ground-truth cliff. Censored stints (541) are `NaN`. |

**Reconciliation Note**: In [`reports/four_model_validation_summary.csv`](file:///C:/Users/radhe/Desktop/deep%20learning/reports/four_model_validation_summary.csv), the column `n_stints` is computed by `n_stints = ("mae_h1", "count")`, which evaluates to 903. While `mae_h1` indeed evaluates all 903 stints, the effective sample size for `pwe_cliff` is strictly the **362 uncensored stints**.

---

## 7. Column & Feature Utilization Classification

All 14 columns in [`data/processed/train_laps.parquet`](file:///C:/Users/radhe/Desktop/deep%20learning/data/processed/train_laps.parquet) were traced to establish their exact functional roles:

| Column Name | Pipeline Role | Functional Classification | Directly Affects Model Output? |
| :--- | :--- | :--- | :---: |
| `race_id` | Split partitioning & stint ID prefix | Split / Leakage Control | No (Metadata/Control) |
| `track_id` | Categorical 4-d embedding & baseline curve index | Direct Model Feature + Baseline | **Yes** |
| `driver` | Stint grouping & dashboard selector | Grouping / Display Metadata | No (Metadata) |
| `stint_id` | Sequence window grouping & cliff labeling | Grouping / Split Control | No (Structural Key) |
| `stint_lap` | Sequencing index ($1 \dots n$) & cutoff slicing | Sequencing / Causal Control | **Yes** |
| `tyre_age` | Numeric input feature #1 & quadratic wear input | Direct Model Feature | **Yes** |
| `compound` | Categorical 3-d embedding & baseline curve index | Direct Model Feature + Baseline | **Yes** |
| `lap_number` | Derives `lap_fraction = lap_number / total_laps` | Derived Model Feature | **Yes** |
| `total_laps` | Derives `lap_fraction` and track-scaled fuel penalty | Derived Feature + Physics | **Yes** |
| `lap_time_s` | Raw physical lap time before fuel correction | Target Derivation Input | **Yes** |
| `lap_time_fc_s` | Fuel-corrected pace; derives residuals & targets | Target + Derived Features | **Yes** |
| `track_temp_c` | Numeric input feature #4 & `track_temp_delta_c` | Direct Model Feature | **Yes** |
| `track_status` | Numeric input feature #5 (flag status integer) | Direct Model Feature | **Yes** |
| `gap_mask` | Numeric input feature #6 (traffic spike / gap flag) | Direct Model Feature | **Yes** |

**Orphaned Columns**: **Zero**. Every column serves a documented, active purpose.

---

## 8. Data Attrition Reconciliation

From [`reports/attrition.csv`](file:///C:/Users/radhe/Desktop/deep%20learning/reports/attrition.csv), across the full 91-race raw dataset (100,391 raw laps):

- **Raw Telemetry Laps Ingested**: 100,391 laps.
- **Physical Exclusions**:
  - `drop_wet_race`: 12,628 laps (non-dry racing violates dry tyre degradation physics).
  - `drop_lap_one`: 1,574 laps (standing-start non-flying laps).
  - `drop_in_out_laps`: 5,395 laps (pit lane speed limiter deltas).
  - `drop_sc_vsc`: 3,858 laps (Safety Car / VSC mandatory delta pacing).
  - `drop_inaccurate_deleted`: 1,021 laps (track limits or transponder errors).
  - `drop_missing_fields`: 479 laps (null timestamps, missing tyre age).
  - `drop_short_stints`: 2,631 laps (stints $< 3$ clean laps cannot establish an early-stint baseline).
- **Clean Laps Retained**: 72,805 laps total.
  - Permitted Train (2022–2023): **33,269 laps**
  - Permitted Val (2024): **20,028 laps**
  - Locked Test (2025): **19,508 laps** (untouched under Rule R3).
- **Flagged Spikes**: 417 laps flagged as temporary traffic anomalies; **0 rows dropped**, retained with `gap_mask = 1`.

---

## 9. Scientific Interpretation & Answers to Key Questions

### 1. Are all eligible source laps used as input at least once?
**Yes**. Every single clean lap ($t = 1 \dots n$) in each stint acts as the terminal endpoint of a sliding sequence window. In training, all 33,269 clean laps appear as sequence endpoints, and on average appear $7.63$ times across overlapping sliding windows.

### 2. Are all eligible laps used as training targets where the existing target definition permits?
**Yes**. Every lap from $t = 2 \dots n$ within each stint serves as a target for preceding laps within the 20-lap horizon. The only laps that cannot be targets are lap 1 of each stint (no preceding lap exists to predict it) and laps beyond the end of the stint (which do not exist).

### 3. Are all valid examples consumed during each training epoch?
**Yes**. PyTorch DataLoader with `batch_size = 64` and `drop_last = False` consumes all 33,269 training examples across 520 batches per epoch without skipping or subsampling.

### 4. What percentage of generated target slots actually contributes to the loss?
**49.35% in training** (328,350 out of 665,380 slots). The remaining 50.65% (337,030 slots) are masked out with NaNs because stints terminate before horizon 20. In validation, **55.66%** of target slots are valid (222,934 out of 400,560 slots).

### 5. Which losses of usable data are intentional, and which are accidental?
- **Intentional Physical Reductions**:
  - End-of-stint target truncation (50.65% train, 44.34% val): Stints end when cars pit; future laps do not exist on that tyre set.
  - Final-lap zero-target examples (1,756 train examples, 903 val examples): The final lap of each stint has no future laps, producing zero loss gradient.
  - Attrition filtering: Wet races, standing starts, pit stops, and SC laps are removed to prevent unphysical pace corruption.
  - Uncensored cliff restriction ($N = 362$ vs $N = 903$): Stints without a cliff cannot evaluate cliff timing error.
- **Accidental Reductions**: **Zero**. There is no unintentional data loss from broken joins, unhandled NaNs, off-by-one indexing, or file-loading crashes.

### 6. Does the current pipeline make maximal use of the data under its existing causal, chronological, and per-stint rules?
**Yes**. Given the causal constraint (no lookahead), temporal split (2022–2023 train, 2024 val), and physical stint boundaries (no cross-stint windowing), the pipeline extracts the maximal number of causal sliding windows (1 per clean lap).

### 7. Is the phrase "100% dataset utilization" technically defensible?
**No, not without precise qualification**.
- It is true that **100.0% of clean source laps are utilized as input sequence endpoints** (33,269 / 33,269 train laps).
- However, **only 49.35% of generated training target slots contain valid supervisory signal** (328,350 / 665,380 slots), and **5.28% of training windows (1,756 windows) contain zero valid target values**.
- A precise, scientifically defensible statement is:
  > *"The pipeline extracts 100% of causally valid sliding windows from permitted clean laps, of which 49.35% of multi-horizon target slots contain valid supervisory signal within physical stint boundaries."*

### 8. Can any stronger conclusion be reached without changing the frozen project?
**No**. Any attempt to artificially increase target slot density (such as stitching telemetry across pit stops or imputing laps past stint termination) would violate Formula 1 tyre wear physics. Under the frozen architecture and causal rules, data utilization is maximal and sound.

---

## 10. Audit Summary & Conclusion

- **Eligible Clean Laps**: 33,269 (Train) / 20,028 (Val).
- **Generated Sequence Windows**: 33,269 (Train) / 20,028 (Val).
- **Valid Supervisory Target Slots**: 328,350 (Train) / 222,934 (Val).
- **Effective Validation Sample Sizes**: 903 stints ($h=1$), 836 stints ($h=5$), 684 stints ($h=10$), 362 stints (Cliff PWE).
- **Rule R3 Compliance**: Strictly preserved. The locked 2025 test dataset was never accessed.
- **Verdict**: **EFFECTIVE USE VERIFIED WITH INTENTIONAL LIMITATIONS**.
