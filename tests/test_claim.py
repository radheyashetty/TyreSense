import pytest
import numpy as np
import pandas as pd
from src.claim_eval import compute_pwe, run_paired_bootstrap, simulate_modes_at_lap


def test_compute_pwe_valid():
    assert compute_pwe(20.0, 18.0) == 2.0
    assert compute_pwe(15.5, 17.5) == 2.0
    assert np.isnan(compute_pwe(None, 18.0))
    assert np.isnan(compute_pwe(20.0, None))


def test_run_paired_bootstrap():
    diffs = np.array([1.0, 2.0, -0.5, 1.5, 0.0, 2.5])
    mean_diff, std_diff, (ci_low, ci_high) = run_paired_bootstrap(diffs, n_boot=500, seed=42)
    assert ci_low <= mean_diff <= ci_high
    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)


def test_simulate_modes_insufficient_laps():
    df = pd.DataFrame({
        "track_id": ["monza", "monza"],
        "compound": ["HARD", "HARD"],
        "stint_lap": [1, 2],
        "tyre_age": [1, 2],
        "lap_time_fc_s": [85.0, 85.2]
    })
    res = simulate_modes_at_lap(df, {}, (0.05, 0.001), target_lap=5)
    assert res is None
