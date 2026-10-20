import time
import pandas as pd
from app import (
    load_app_data,
    get_stint_subset,
    forecast_future_laps,
    compute_remaining_life,
    plot_stint_replay,
)


def test_app_first_render_and_data_load_time():
    """Verify first data load and render benchmark from cache is < 3 seconds."""
    start_time = time.time()
    laps, stints, curves, global_curve = load_app_data()
    duration = time.time() - start_time
    print(f"Data load time: {duration:.3f} s")
    assert duration < 3.0, f"App first load took {duration:.2f}s (exceeds 3s target)"
    assert not laps.empty
    assert not stints.empty


def test_forecast_causality_and_step_render_time():
    """Verify forecast sees only laps <= current lap and generates replay step in < 1s."""
    laps, _, curves, global_curve = load_app_data()
    sample_stint = laps["stint_id"].iloc[0]
    full_stint = get_stint_subset(laps, sample_stint)

    curr_lap = 5
    observed = full_stint[full_stint["stint_lap"] <= curr_lap]
    assert int(observed["stint_lap"].max()) == curr_lap

    step_start = time.time()
    fut_laps, fut_times, base_time = forecast_future_laps(observed, curves, global_curve)
    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap, base_time)
    fig = plot_stint_replay(observed, fut_laps, fut_times, base_time)
    step_duration = time.time() - step_start

    print(f"Single step render time: {step_duration:.3f} s")
    assert step_duration < 1.0, f"Step execution took {step_duration:.2f}s (exceeds 1s limit)"
    assert len(fut_laps) == 20
    assert len(fut_times) == 20
    assert isinstance(rem_life, str)
    assert fig is not None


def test_uncertainty_band_hidden_before_lap_3():
    """Verify uncertainty band is hidden at lap 1 and 2, and rendered at lap 3."""
    laps, _, curves, global_curve = load_app_data()
    sample_stint = laps["stint_id"].iloc[0]
    full_stint = get_stint_subset(laps, sample_stint)

    # Lap 2: band should not be in figure traces
    obs_lap2 = full_stint[full_stint["stint_lap"] <= 2]
    fut_laps, fut_times, base_time = forecast_future_laps(obs_lap2, curves, global_curve)
    fig_lap2 = plot_stint_replay(obs_lap2, fut_laps, fut_times, base_time, curr_lap=2)
    names_lap2 = [t.name for t in fig_lap2.data if t.name is not None]
    assert "80% Uncertainty Band" not in names_lap2

    # Lap 3: band should be present in figure traces
    obs_lap3 = full_stint[full_stint["stint_lap"] <= 3]
    fut_laps3, fut_times3, base_time3 = forecast_future_laps(obs_lap3, curves, global_curve)
    fig_lap3 = plot_stint_replay(obs_lap3, fut_laps3, fut_times3, base_time3, curr_lap=3)
    names_lap3 = [t.name for t in fig_lap3.data if t.name is not None]
    assert "80% Uncertainty Band" in names_lap3
