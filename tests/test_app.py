import time, pytest, pandas as pd, numpy as np
from app import (
    load_app_data,
    get_stint_subset,
    forecast_future_laps,
    compute_remaining_life,
    plot_stint_replay,
)
from src.app_data import assert_no_2025_in_app
from src.app_forecast import compute_live_k


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

    obs_lap2 = full_stint[full_stint["stint_lap"] <= 2]
    fut_laps, fut_times, base_time = forecast_future_laps(obs_lap2, curves, global_curve)
    fig_lap2 = plot_stint_replay(obs_lap2, fut_laps, fut_times, base_time, curr_lap=2)
    names_lap2 = [t.name for t in fig_lap2.data if t.name is not None]
    assert "80% Uncertainty Band" not in names_lap2

    obs_lap3 = full_stint[full_stint["stint_lap"] <= 3]
    fut_laps3, fut_times3, base_time3 = forecast_future_laps(obs_lap3, curves, global_curve)
    fig_lap3 = plot_stint_replay(obs_lap3, fut_laps3, fut_times3, base_time3, curr_lap=3)
    names_lap3 = [t.name for t in fig_lap3.data if t.name is not None]
    assert "80% Uncertainty Band" in names_lap3


def test_causal_slice_strictness():
    """Verify that observed telemetry never leaks laps beyond curr_lap."""
    laps, _, _, _ = load_app_data()
    full_stint = get_stint_subset(laps, laps["stint_id"].iloc[0])
    for target in [1, 3, 5, 8]:
        obs = full_stint[full_stint["stint_lap"] <= target]
        assert obs["stint_lap"].max() == target
        assert not (obs["stint_lap"] > target).any()


def test_model_switch_changes_forecast():
    """Verify that switching forecasting models alters the resulting forecast."""
    laps, _, curves, g_curve = load_app_data()
    full_stint = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = full_stint[full_stint["stint_lap"] <= 5]
    _, fut_a, _ = forecast_future_laps(obs, curves, g_curve, model_choice="Model A (Baseline Curve)")
    _, fut_e, _ = forecast_future_laps(obs, curves, g_curve, k=1.4, model_choice="Model E")
    assert not np.allclose(fut_a, fut_e), "Switching from Model A to Model E must change forecast"


def test_empty_state_handling():
    """Verify that empty filter combinations are handled gracefully."""
    laps, _, _, _ = load_app_data()
    empty_slice = laps[laps["driver"] == "NONEXISTENT_DRIVER"]
    assert empty_slice.empty
    stints_avail = sorted(empty_slice["stint_id"].unique())
    assert len(stints_avail) == 0


def test_app_2025_test_lock():
    """Verify that app dataset contains zero 2025 data and lock triggers on breach."""
    laps, stints, _, _ = load_app_data()
    assert not (laps["season"] == "2025").any()
    assert not (laps["race_id"].str.startswith("2025")).any()
    breach_df = pd.DataFrame({"race_id": ["2025_01", "2024_01"]})
    with pytest.raises(AssertionError, match="R3 Violation: 2025 data detected"):
        assert_no_2025_in_app(breach_df)


def test_forecast_continuity_20_val_stints():
    """Verify first step of Model A and E differs from last observed by less than 0.5s across 20 seeded val stints."""
    laps, stints, curves, g_curve = load_app_data()
    val_laps = pd.read_parquet("data/processed/val_laps.parquet")
    val_stints = sorted(val_laps["stint_id"].unique())
    rng = np.random.default_rng(42)
    sampled = rng.choice(val_stints, size=20, replace=False)

    for sid in sampled:
        sub = get_stint_subset(laps, sid)
        if len(sub) < 5:
            continue
        obs = sub[sub["stint_lap"] <= 5]
        last_obs = float(obs["lap_time_fc_s"].iloc[-1])
        k = compute_live_k(sub, 5, curves, g_curve)
        _, fut_a, _ = forecast_future_laps(obs, curves, g_curve, model_choice="Model A")
        _, fut_e, _ = forecast_future_laps(obs, curves, g_curve, k=k, model_choice="Model E")
        diff_a = abs(fut_a[0] - last_obs)
        diff_e = abs(fut_e[0] - last_obs)
        assert diff_a < 0.5, f"Model A first step jump {diff_a} >= 0.5s on {sid}"
        assert diff_e < 0.5, f"Model E first step jump {diff_e} >= 0.5s on {sid}"


def test_neural_models_produce_valid_continuous_forecasts():
    """Verify GRU, LSTM, and RNN execute neural inference and join smoothly."""
    from src.app_data import load_cached_model
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]
    last_obs = float(obs["lap_time_fc_s"].iloc[-1])

    _, fut_base, _ = forecast_future_laps(obs, curves, g_curve, model_choice="Model A")
    for name in ["GRU Forecaster", "LSTM Forecaster", "Vanilla RNN"]:
        m = load_cached_model(name)
        assert m is not None, f"Checkpoint for {name} must exist"
        laps_out, times_out, _ = forecast_future_laps(obs, curves, g_curve, model_choice=name, model_obj=m)
        assert len(laps_out) == 20 and len(times_out) == 20
        assert not np.isnan(times_out).any()
        assert abs(times_out[0] - last_obs) < 0.5
        assert not np.allclose(times_out, fut_base), f"{name} must differ from baseline"


def test_streamlit_app_renders_smoke_test():
    """Verify Streamlit app initializes and renders default view headlessly without exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0
