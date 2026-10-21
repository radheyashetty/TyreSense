# TYRESENSE FINAL FIX + UI + CLEANUP RUN REPORT

**Date & Time**: 2026-10-21T08:48:00+05:30  
**Branch**: `overnight-run`  
**Base Commit**: `55a5fc7` (tagged `pre-run2`)  
**Experiment Python Interpreter**: `C:\Python314\python.exe` (Python 3.14.3)  
**Test Runner Python Interpreter**: `C:\Users\radhe\AppData\Local\Programs\Python\Python310\python.exe` (Python 3.10.11)  

---

## Part 1: Verify and Rename

### 1. Model Input Dimensions and Parameter Counts
Inspected recurrent architectures as built by `src/models.py`:
- **GRU**: Total parameters = 5,802, Input dimension = 15, `weight_ih_l0` shape = `torch.Size([96, 15])`
- **LSTM**: Total parameters = 5,770, Input dimension = 15, `weight_ih_l0` shape = `torch.Size([108, 15])`
- **RNN**: Total parameters = 5,864, Input dimension = 15, `weight_ih_l0` shape = `torch.Size([58, 15])`

Input dimension breakdown:
- 8 numeric features (`tyre_age`, `lap_fraction`, `relative_pace`, `track_temp`, `track_status`, `gap_mask`, `track_temp_delta_c`, `curr_residual`)
- 4 track embedding dimensions (`nn.Embedding(50, 4)`)
- 3 compound embedding dimensions (`nn.Embedding(10, 3)`)
- Total recurrent input size: $8 + 4 + 3 = 15$.

Saved Checkpoint Verification (`runs/gru_main/checkpoint.pt`):
- Loaded weights file: `runs/gru_main/checkpoint.pt`
- First-layer weight `rnn.weight_ih_l0` shape: `torch.Size([96, 15])`
- **Confirmed**: Model was trained with all 8 numeric features wired.

### 2. Checkpoint Directory Renaming
- Renamed `runs/final_gru` to `runs/gru_main` via `git mv`.
- Updated all references across:
  - `app.py:28`
  - `src/runner_part_c.py:94-99`
  - `src/runner_part_e.py:132`
  - `reports/overnight/STATE.json:21-22`
  - `reports/overnight/results_hashes.csv:18`
  - `reports/overnight/STAGE_REPORT.md:101, 212`

### 3. Environment and Interpreter Verification
- **Experiment Interpreter**: Python 3.14.3 (`C:\Python314\python.exe`)
- **Pytest Interpreter**: Python 3.10.11 (`C:\Users\radhe\AppData\Local\Programs\Python\Python310\python.exe`)
- Ran `python -m pytest` under Python 3.14.3:
  ```text
  platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0
  collected 126 items
  107 passed, 19 skipped, 1 warning in 18.92s
  ```

### 4. Git Diff Against `pre-final-run` for `tests/`
Command: `git diff pre-final-run -- tests/`
- Lines removed in `tests/test_split.py`: 23 lines condensed using `make_mock_row` helper function to satisfy `AGENT_RULES.md` (<=30 lines per function rule).
- Lines removed in `tests/test_live.py`: Line 51 (`assert abs(k - true_k) < abs(1.0 - true_k)`) was removed in previous investigation. Flagged for restoration in Part 3 with one-step-ahead update rule.
- No other test deletions.
