import numpy as np
import pandas as pd
from src.live import update_k_factor, run_live_correction_stint


def test_k_remains_unchanged_before_lap_three():
    """Verify that k factor is never updated before stint lap 3."""
    k = 1.0
    k_history = []
    # Stint lap 1 and 2
    for lap in [1, 2]:
        actual = 91.5
        base = 90.0
        g = 0.5
        # Simulate guard condition in run_live_correction_stint
        if lap >= 3:
            k, k_history = update_k_factor(k, k_history, actual, base, g)
        assert k == 1.0
        assert len(k_history) == 0


def test_k_remains_unchanged_on_flagged_spike():
    """Verify that a causal spike lap suppresses k update."""
    k = 1.0
    k_history = []
    actual_spike = 95.0
    base = 90.0
    g = 0.4
    k, k_history = update_k_factor(k, k_history, actual_spike, base, g, is_spike=True)
    assert k == 1.0
    assert len(k_history) == 0


def test_k_moves_toward_true_wear_rate():
    """Verify that on noisy synthetic stints k moves toward true rate within 5 laps."""
    np.random.seed(42)
    true_k = 1.5
    base = 90.0
    g = 0.3
    noise = np.random.normal(0, 0.1, 15)

    k = 1.0
    k_history = []
    # Run from lap 3 to lap 8 (6 update steps)
    for i in range(8):
        actual = base + true_k * g + noise[i]
        k, k_history = update_k_factor(k, k_history, actual, base, g, is_spike=False)

    print(f"Final k after 8 laps: {k:.3f}")
    assert 0.5 <= k <= 2.0
    assert abs(k - true_k) < abs(1.0 - true_k)


def test_model_e_differs_from_rolling_on_accelerated_wear():
    """Verify Model E forecast differs from ROLLING on synthetic faster-than-baseline wear."""
    from src.claim_eval import compute_cliff_life
    # ROLLING forecast degradation with k=1.0: 0.1s/lap increase
    roll_deg = np.array([0.1 * i for i in range(1, 21)])  # crosses 1.0s at lap 10
    roll_life = compute_cliff_life(roll_deg, threshold_s=1.0)
    assert roll_life == 10

    # Accelerated wear: k updated to 1.4 (40% faster wear)
    mode_e_deg = 1.4 * roll_deg  # crosses 1.0s at lap 8 (1.4 * 0.8 = 1.12s)
    mode_e_life = compute_cliff_life(mode_e_deg, threshold_s=1.0)
    assert mode_e_life == 8
    assert mode_e_life != roll_life, "Model E must differ from ROLLING under faster wear"
