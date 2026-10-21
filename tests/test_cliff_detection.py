import numpy as np
from src.app_forecast import compute_cliff_lap_int, compute_remaining_life, forecast_future_laps
from src.live import compute_cliff_life


def test_cliff_detection_no_crossing():
    """Verify None and '> 20 laps' when degradation never reaches threshold."""
    fut_laps = np.arange(6, 26)
    fut_times = np.full(20, 90.5)
    base_time = 90.0

    cliff = compute_cliff_lap_int(fut_laps, fut_times, base_time, threshold=1.0)
    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap=5, baseline_time=base_time)
    assert cliff is None
    assert rem_life == "> 20 laps"


def test_cliff_detection_isolated_crossing_ignored():
    """Verify single isolated spike does not trigger cliff without second consecutive lap."""
    fut_laps = np.arange(6, 26)
    fut_times = np.full(20, 90.5)
    fut_times[3] = 92.0  # lap 9 has deg = 2.0s, but lap 10 is 90.5s (deg = 0.5s)
    base_time = 90.0

    cliff = compute_cliff_lap_int(fut_laps, fut_times, base_time, threshold=1.0)
    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap=5, baseline_time=base_time)
    assert cliff is None
    assert rem_life == "> 20 laps"


def test_cliff_detection_two_consecutive_crossings():
    """Verify cliff fires on first lap of first two consecutive qualifying laps."""
    fut_laps = np.arange(6, 26)
    fut_times = np.full(20, 90.5)
    fut_times[4] = 91.2  # lap 10: deg = 1.2s
    fut_times[5] = 91.4  # lap 11: deg = 1.4s
    base_time = 90.0

    cliff = compute_cliff_lap_int(fut_laps, fut_times, base_time, threshold=1.0)
    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap=5, baseline_time=base_time)
    assert cliff == 10
    assert rem_life == "5 laps"


def test_cliff_detection_boundary_conditions():
    """Verify crossing at the very end of the horizon requires 2 valid consecutive points."""
    fut_laps = np.arange(6, 26)
    base_time = 90.0

    times_single_last = np.full(20, 90.5)
    times_single_last[19] = 91.5
    assert compute_cliff_lap_int(fut_laps, times_single_last, base_time) is None

    times_pair_last = np.full(20, 90.5)
    times_pair_last[18] = 91.2
    times_pair_last[19] = 91.5
    cliff = compute_cliff_lap_int(fut_laps, times_pair_last, base_time)
    rem = compute_remaining_life(fut_laps, times_pair_last, curr_lap=5, baseline_time=base_time)
    assert cliff == 24
    assert rem == "19 laps"


def test_live_compute_cliff_life_consecutive():
    """Verify compute_cliff_life in src/live.py requires 2 consecutive laps."""
    deg_single = np.array([0.2, 0.4, 1.3, 0.5, 0.6])
    assert compute_cliff_life(deg_single, threshold_s=1.0) is None

    deg_pair = np.array([0.2, 0.4, 1.1, 1.2, 0.6])
    assert compute_cliff_life(deg_pair, threshold_s=1.0) == 3


def test_direct_reconstruction_mode_parameter():
    """Verify mode parameter toggles direct reconstruction vs continuity adjustment."""
    from src.app_data import load_app_data, load_cached_model, get_stint_subset
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]
    m = load_cached_model("GRU Forecaster")

    _, y_cont, _ = forecast_future_laps(obs, curves, g_curve, model_choice="GRU Forecaster", model_obj=m, mode="continuity_adjusted")
    _, y_dir, _ = forecast_future_laps(obs, curves, g_curve, model_choice="GRU Forecaster", model_obj=m, mode="direct_reconstruction")
    assert len(y_cont) == 20 and len(y_dir) == 20
    assert not np.isnan(y_cont).any() and not np.isnan(y_dir).any()
