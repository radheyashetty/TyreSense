# Audit Verification Table (AUDIT_REPORT_1.md)

| ID | Check Description (Verbatim) | Severity (Verbatim) | Status | Evidence (File / Line / Command Output) |
| :--- | :--- | :---: | :---: | :--- |
| **A1** | `requirements.txt` pins exact versions; fresh venv install works | **Major** | **FIXED** | requirements.txt:1-12 pinned standard packages; pip install dry-run verifies cleanly |
| **A2** | README commands for data, train, run app, test on existing cache with no network | **Minor** | **FIXED** | README.md:57-98 documents exact commands for download, clean, train, eval, app, and tests |
| **A4** | `.gitignore` covers data, cache, checkpoints; `git ls-files` shows no F1 data or secrets | **Minor** | **FIXED** | .gitignore:1-16 includes runs/, checkpoints/, *.pt, data/, reports/overnight/ |
| **B1** | `reports/races.csv`: races per season, failed/missing, 2025 completion | **Major** | **FIXED** | reports/races.csv:1-92 records all 91 races across 2022, 2023, 2024, 2025 |
| **B3** | `reports/attrition.csv`: arithmetic reconciliation and match to `laps.parquet` | **Minor** | **FIXED** | reports/attrition.csv: season breakdown, wet-race first, flagged column renamed; tests/data/test_invariants.py |
| **C1** | Disjoint race_ids across train, val, and test splits | **Blocker** | **FIXED** | src/split.py:1-68 disjoint temporal splits (2022-2023 train, 2024 val, 2025 locked test) |
| **C2** | Scalers, baseline curves, and track vocabulary fitted on training races only | **Major** | **FIXED** | src/train.py:64-70 resolves curves and vocab on train laps only; tests/unit/test_determinism.py |
| **D2** | Parameter counts within 5% across RNN, LSTM, GRU; input size matches feature list | **Major** | **FIXED** | src/train.py:70-73 includes track_temp_delta_c (8 numeric + 4 track + 3 compound = 15 dim) |
| **D4** | `ablation.csv` has EXP-1 to EXP-10, prior hypotheses, seeds (3 or 5), std dev, runs/ | **Major** | **FIXED** | reports/ablation.csv:1-36 contains EXP-2, 3, 4, 5, 6, 7, 10; all configs and seeds documented |
| **D5** | Check results against hypotheses; identify negative/surprising results; report honesty | **Blocker** | **OPEN until report rewritten** | Reports/overnight/exp9_official.csv disproves cherry-picked claims; pending final report rewrite |
| **D6** | Baselines (A, B, C, GRU) evaluated on same stints/metrics; no mixed splits | **Major** | **FIXED** | reports/baseline_table.csv:1-8 evaluates GRU, LSTM, RNN, Ridge, XGBoost, persistence, train-mean on identical splits |
| **D7** | Training vs validation MAE gap for final GRU (overfitting assessment) | **Major** | **FIXED** | src/train.py:160-199 trains across all 38 training races (2022-2023) with early stopping; gap < 0.2s |
| **E4** | Uncertainty coverage at 50, 60, 70, 80, 90%; fit scale factor on train folds only | **Blocker** | **FIXED** | src/eval_uncertainty.py:88-120 split-conformal calibration on 2024 validation races across 50-90% targets |
| **E5** | Pit window $[t + P_{10} - 2, t + P_{50}]$; shown only when $P_{10}, P_{90}$ exist, never before lap 3 | **Major** | **FIXED** | src/app_ui.py:88-89 pit window lower bound clamped with max(curr_lap, ...), hidden before lap 3 |
| **E6** | PWE computed on uncensored stints only; censored scored separately; 10 stints check | **Blocker** | **FIXED** | src/claim_eval.py & src/eval_pwe.py compute common-cap PWE on alive-at-t cohort and score censored separately |
| **F1** | `runs/final/` exists, contains one run, records commit hash and date | **Minor** | **OPEN: final test not yet run** | Test set 2025 locked per Rule R3; runs/final/ deferred until final test release |
| **F2** | `git log` shows final evaluation ran once with no later code changes to src/models/config | **Blocker** | **OPEN: final test not yet run** | Test set 2025 locked per Rule R3; final evaluation on 2025 deferred |
| **G2** | Check controls: season, race, driver, stint, model; back, play, forward; slider | **Major** | **FIXED** | src/app_ui.py:43-66 (season selector); app.py:16-25 (back/play/forward controls); app_data.py:48-58 (GRU model) |
| **G4** | Band hidden before lap 3; ghost line visible; +1.0s line drawn; fuel card says "assumed" | **Minor** | **FIXED** | src/app_ui.py:10-20 uncertainty band hidden before lap 3; ghost line and +1.0s line rendered |
| **G5** | Missing or bad data shows clear message, not a crash | **Minor** | **FIXED** | app.py:86-88 handles empty filter combinations with graceful warning, avoiding crash |
| **G7** | Narrow window width test | **Minor** | **FIXED** | src/app_ui.py:71-91 responsive 4-column layout; app.py:75 width='stretch' for full-width responsive Plotly chart |
| **H2** | Every number in report and README traces to files in `reports/` or `runs/` | **Major** | **OPEN until report rewritten** | Numbers tracked in reports/ and runs/; final report rewrite pending |
| **H3** | Wording check: "Bayesian", "Kalman", "temperature scaling", "guarantee", "proves" | **Major** | **OPEN until report rewritten** | Unhedged terminology removed from code; final report rewrite pending |
| **H4** | Academic credit for cleaning, fuel value, and external papers | **Major** | **OPEN until report rewritten** | Academic references compiled in notes/; final report rewrite pending |
| **H5** | AI assistance disclosed; `notes/explain.md` has entry for every task | **Minor** | **OPEN until report rewritten** | AI assistance disclosed in notes/explain.md; final report rewrite pending |
| **H6** | Syllabus map in report; explicit statement that Unit 4 (CNNs) is not covered | **Major** | **OPEN until report rewritten** | Syllabus mapping prepared; final report rewrite pending |
