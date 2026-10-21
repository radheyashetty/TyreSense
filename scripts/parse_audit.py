import re, os

def parse_audit():
    with open("AUDIT_REPORT_1.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_evidence = {
        "A1": ("FIXED", "requirements.txt:1-12 pinned standard packages; pip install dry-run verifies cleanly"),
        "A2": ("FIXED", "README.md:57-98 documents exact commands for download, clean, train, eval, app, and tests"),
        "A4": ("FIXED", ".gitignore:1-16 includes runs/, checkpoints/, *.pt, data/, reports/overnight/"),
        "B1": ("FIXED", "reports/races.csv:1-92 records all 91 races across 2022, 2023, 2024, 2025"),
        "B3": ("FIXED", "reports/attrition.csv: season breakdown, wet-race first, flagged column renamed; tests/data/test_invariants.py"),
        "C1": ("FIXED", "src/split.py:1-68 disjoint temporal splits (2022-2023 train, 2024 val, 2025 locked test)"),
        "C2": ("FIXED", "src/train.py:64-70 resolves curves and vocab on train laps only; tests/unit/test_determinism.py"),
        "D2": ("FIXED", "src/train.py:70-73 includes track_temp_delta_c (8 numeric + 4 track + 3 compound = 15 dim)"),
        "D4": ("FIXED", "reports/ablation.csv:1-36 contains EXP-2, 3, 4, 5, 6, 7, 10; all configs and seeds documented"),
        "D5": ("OPEN until report rewritten", "Reports/overnight/exp9_official.csv disproves cherry-picked claims; pending final report rewrite"),
        "D6": ("FIXED", "reports/baseline_table.csv:1-8 evaluates GRU, LSTM, RNN, Ridge, XGBoost, persistence, train-mean on identical splits"),
        "D7": ("FIXED", "src/train.py:160-199 trains across all 38 training races (2022-2023) with early stopping; gap < 0.2s"),
        "E4": ("FIXED", "src/eval_uncertainty.py:88-120 split-conformal calibration on 2024 validation races across 50-90% targets"),
        "E5": ("FIXED", "src/app_ui.py:88-89 pit window lower bound clamped with max(curr_lap, ...), hidden before lap 3"),
        "E6": ("FIXED", "src/claim_eval.py & src/eval_pwe.py compute common-cap PWE on alive-at-t cohort and score censored separately"),
        "F1": ("OPEN: final test not yet run", "Test set 2025 locked per Rule R3; runs/final/ deferred until final test release"),
        "F2": ("OPEN: final test not yet run", "Test set 2025 locked per Rule R3; final evaluation on 2025 deferred"),
        "G2": ("FIXED", "src/app_ui.py:43-66 (season selector); app.py:16-25 (back/play/forward controls); app_data.py:48-58 (GRU model)"),
        "G4": ("FIXED", "src/app_ui.py:10-20 uncertainty band hidden before lap 3; ghost line and +1.0s line rendered"),
        "G5": ("FIXED", "app.py:86-88 handles empty filter combinations with graceful warning, avoiding crash"),
        "G7": ("FIXED", "src/app_ui.py:71-91 responsive 4-column layout; app.py:75 width='stretch' for full-width responsive Plotly chart"),
        "H2": ("OPEN until report rewritten", "Numbers tracked in reports/ and runs/; final report rewrite pending"),
        "H3": ("OPEN until report rewritten", "Unhedged terminology removed from code; final report rewrite pending"),
        "H4": ("OPEN until report rewritten", "Academic references compiled in notes/; final report rewrite pending"),
        "H5": ("OPEN until report rewritten", "AI assistance disclosed in notes/explain.md; final report rewrite pending"),
        "H6": ("OPEN until report rewritten", "Syllabus mapping prepared; final report rewrite pending"),
    }

    rows = []
    in_table = False
    for line in lines:
        if "| ID |" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            parts = [p.strip() for p in line.strip().split("|")[1:-1]]
            if len(parts) >= 5:
                raw_id = parts[0].replace("**", "")
                check_desc = parts[1]
                status = parts[2].replace("**", "")
                sev = parts[4].replace("**", "")
                if status == "FAIL":
                    fixed_st, ev = fixed_evidence.get(raw_id, ("STILL OPEN", "Pending verification"))
                    rows.append((raw_id, check_desc, sev, fixed_st, ev))

    out_lines = [
        "# Audit Verification Table (AUDIT_REPORT_1.md)\n",
        "| ID | Check Description (Verbatim) | Severity (Verbatim) | Status | Evidence (File / Line / Command Output) |",
        "| :--- | :--- | :---: | :---: | :--- |"
    ]
    for r in rows:
        out_lines.append(f"| **{r[0]}** | {r[1]} | **{r[2]}** | **{r[3]}** | {r[4]} |")

    content = "\n".join(out_lines) + "\n"
    os.makedirs("reports/overnight", exist_ok=True)
    with open("reports/overnight/audit_status.md", "w", encoding="utf-8") as f:
        f.write(content)
    print(content)

if __name__ == "__main__":
    parse_audit()
