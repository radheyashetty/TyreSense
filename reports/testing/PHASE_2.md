# Phase 2 Summary: Centralized Track Data System

## 1. Overview
Phase 2 replaced hardcoded track assumptions across the repository with a single source of truth: `config/tracks.yaml` managed by `src/tracks.py`. All 25 Formula 1 circuits appearing on the 2022-2025 calendars are defined, verified, and mapped to canonical lowercase track IDs.

In addition:
- `reports/races.csv` now contains a mandatory `track_id` column populated via `canonical_track_id(event_name)` (0 null values across all 93 scheduled/run race rows).
- `src/download.py` safely intercepts unknown circuits: instead of crashing, it logs the race with status `unknown_track`, appends the event to `reports/tracks_pending.csv`, and continues.
- `reports/testing/track_coverage.csv` records the 25-circuit by 4-season presence matrix.

---

## 2. Circuit Verification Status
- Total circuits defined: 25
- Circuits with verified length: 25 (100%)
- Circuits with unverified length: 0
- Circuits with null length: 0
- Unknown circuit names in `reports/races.csv`: 0

### Calendar Circuit Counts
- **2022**: 22 circuits
- **2023**: 23 circuits (22 held; Emilia Romagna GP at Imola cancelled due to flooding)
- **2024**: 24 circuits
- **2025**: 24 circuits

### Generalization & Out-of-Distribution Warning
- **Circuit seen ONLY in validation/test seasons (2024-2025)**: `shanghai` (Chinese Grand Prix).
  - Shanghai returned to the calendar in 2024 after an absence since 2019. Any model trained exclusively on 2022-2023 data will encounter Shanghai as a zero-shot circuit.

---

## 3. How to Add a New Circuit (4 Steps)

### Step 1: Register Track ID and Metadata
Run the CLI command with the canonical lowercase `track_id`, official name, and country:
```bash
python -m src.tracks add-track madrid "Circuito de Madrid Jarama" --country Spain
```

### Step 2: Add Known Aliases
Register all naming variations used by FastF1, OpenF1, Ergast/Jolpica, or FIA timing screens:
```bash
python -m src.tracks add-alias madrid "Madrid Grand Prix"
python -m src.tracks add-alias madrid "Spanish Grand Prix - Madrid"
```
*Note: Duplicate aliases across different tracks are strictly rejected.*

### Step 3: Record Verified Lap Distance
Look up the official lap distance on Formula1.com or the FIA technical circuit map and register it:
```bash
python -m src.tracks set-length madrid 5.474 --source "https://www.formula1.com/en/racing/2026/madrid/circuit"
```
If a layout change occurs in a specific season, pass the `--season` argument:
```bash
python -m src.tracks set-length singapore 4.940 --source "https://www.formula1.com/en/racing/2023/singapore/circuit" --season 2023
```

### Step 4: Verify System Status
Verify that no tracks are pending, unverified, or unrecognized:
```bash
python -m src.tracks status
python -m src.tracks scan
```

---

## 4. How to Add a New Season

1. Run download pipeline or schedule discovery for the new season (e.g., `python -m src.download --season 2026`).
2. If any new or renamed circuits appear, `src.download` marks them as `unknown_track` and records them into `reports/tracks_pending.csv`.
3. Check `reports/tracks_pending.csv` for any unmapped events.
4. For each pending circuit, either add it via `python -m src.tracks add-track` (if new) or register the missing alias via `python -m src.tracks add-alias <track_id> "<event_name>"`.
5. Re-run `python -m src.download --season 2026` to download the newly mapped sessions.
6. Regenerate coverage matrix:
   ```bash
   python -c "from src.tracks import coverage_matrix; coverage_matrix().to_csv('reports/testing/track_coverage.csv')"
   ```

---

## 5. Verification & Test Suite
Unit tests in `tests/unit/test_tracks.py` verify:
1. All aliases in `config/tracks.yaml` map unambiguously to their parent `track_id`.
2. No aliases are shared across different circuits.
3. All 91 downloaded raw race parquets resolve to non-empty track IDs matching `reports/races.csv`.
4. All circuit lengths lie in the physically realistic range $[2.5\text{ km}, 8.0\text{ km}]$.
5. Season layout overrides (e.g., Singapore 2023) deviate by $\le 5\%$ from base track lengths.
6. CLI mutations (`add-track`, `add-alias`, `set-length`) work deterministically and reject collisions.
7. Download pipeline handles unrecognized circuits without crashing.
