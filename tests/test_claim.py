import pytest
import numpy as np
import pandas as pd
from src.claim_eval import compute_pwe, compute_pwe_abs, run_paired_bootstrap, simulate_modes_at_lap
from src.eval_pwe import run_stint_boot, run_cluster_boot


def test_compute_pwe_valid():
    assert compute_pwe(20.0, 18.0) == 2.0
    assert compute_pwe(15.5, 17.5) == 2.0
    assert compute_pwe(None, 18.0) == 2.0
    assert np.isnan(compute_pwe(20.0, None))


def test_compute_pwe_abs_common_cap():
    """Verify common absolute horizon capping at lap 22."""
    assert compute_pwe_abs(18.0, 15.0, cap_horizon=22) == 3.0
    assert compute_pwe_abs(25.0, 30.0, cap_horizon=22) == 0.0
    assert compute_pwe_abs(None, 17.0, cap_horizon=22) == 5.0
    assert compute_pwe_abs(10.0, 28.0, cap_horizon=22) == 12.0
    assert np.isnan(compute_pwe_abs(15.0, None, cap_horizon=22))


def test_compute_pwe_capped_and_effective_n():
    """Verify non-crossing forecasts are capped at 20 laps and scored."""
    pwe = compute_pwe(None, 15.0, max_horizon=20)
    assert pwe == 5.0


def test_run_paired_bootstrap():
    diffs = np.array([1.0, 2.0, -0.5, 1.5, 0.0, 2.5])
    mean_diff, std_diff, (ci_low, ci_high) = run_paired_bootstrap(diffs, n_boot=500, seed=42)
    assert ci_low <= mean_diff <= ci_high
    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)


def test_cluster_bootstrap_synthetic_correlated_data():
    """Verify cluster bootstrap correctly inflates CI width when data has intra-cluster correlation."""
    np.random.seed(42)
    n_clusters = 20
    cluster_effects = np.random.normal(0, 3.0, size=n_clusters)
    race_diff_map, all_diffs = {}, []
    for k in range(n_clusters):
        c_diffs = (cluster_effects[k] + np.random.normal(0, 0.5, size=15)).tolist()
        race_diff_map[f"race_{k}"] = c_diffs
        all_diffs.extend(c_diffs)

    _, ci_stint = run_stint_boot(np.array(all_diffs), n_boot=500, seed=42)
    _, ci_cluster = run_cluster_boot(race_diff_map, n_boot=500, seed=42)
    stint_width = ci_stint[1] - ci_stint[0]
    cluster_width = ci_cluster[1] - ci_cluster[0]
    assert cluster_width > stint_width * 2.0


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
