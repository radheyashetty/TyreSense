import numpy as np
import pandas as pd
from src.app_data import load_app_data, get_stint_subset
from src.app_forecast import compute_all_model_forecasts, compute_live_k
from src.app_ui import plot_multi_model_replay


def test_compare_all_models_computes_four_forecasts():
    """Verify compute_all_model_forecasts generates exactly 4 model outputs for the same stint."""
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]
    k_val = compute_live_k(sub, 5, curves, g_curve)

    results = compute_all_model_forecasts(obs, curves, g_curve, k_val)
    assert len(results) == 4
    names = [r["name"] for r in results]
    assert names == ["GRU Forecaster", "LSTM Forecaster", "Vanilla RNN", "Model E (Adaptive)"]

    for r in results:
        assert len(r["fut_laps"]) == 20
        assert len(r["fut_times"]) == 20
        assert not np.isnan(r["fut_times"]).any()
        assert r["cliff_lap"] is None or isinstance(r["cliff_lap"], int)
        assert isinstance(r["rem_life"], str)


def test_recurrent_models_use_independent_checkpoints():
    """Verify GRU, LSTM, and Vanilla RNN produce distinct forecast trajectories."""
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]
    k_val = 1.0

    results = compute_all_model_forecasts(obs, curves, g_curve, k_val)
    times = {r["name"]: r["fut_times"] for r in results}

    diff_gru_lstm = np.max(np.abs(times["GRU Forecaster"] - times["LSTM Forecaster"]))
    diff_gru_rnn = np.max(np.abs(times["GRU Forecaster"] - times["Vanilla RNN"]))
    diff_lstm_rnn = np.max(np.abs(times["LSTM Forecaster"] - times["Vanilla RNN"]))

    assert diff_gru_lstm > 0.05, "GRU and LSTM must differ"
    assert diff_gru_rnn > 0.05, "GRU and Vanilla RNN must differ"
    assert diff_lstm_rnn > 0.05, "LSTM and Vanilla RNN must differ"


def test_model_e_adapts_with_k_multiplier():
    """Verify Model E forecast reflects online wear multiplier k in comparison results."""
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]

    res_k1 = compute_all_model_forecasts(obs, curves, g_curve, k_val=1.0)
    res_k15 = compute_all_model_forecasts(obs, curves, g_curve, k_val=1.5)

    mode_e_k1 = [r for r in res_k1 if r["name"] == "Model E (Adaptive)"][0]["fut_times"]
    mode_e_k15 = [r for r in res_k15 if r["name"] == "Model E (Adaptive)"][0]["fut_times"]

    assert not np.allclose(mode_e_k1, mode_e_k15)
    assert mode_e_k15[-1] > mode_e_k1[-1]


def test_multi_model_replay_plot_traces():
    """Verify plot_multi_model_replay builds valid Plotly figure with all 4 model traces."""
    laps, _, curves, g_curve = load_app_data()
    sub = get_stint_subset(laps, laps["stint_id"].iloc[0])
    obs = sub[sub["stint_lap"] <= 5]
    k_val = compute_live_k(sub, 5, curves, g_curve)
    results = compute_all_model_forecasts(obs, curves, g_curve, k_val)

    fig = plot_multi_model_replay(obs, results, base_time=float(obs["lap_time_fc_s"].iloc[0]))
    trace_names = [t.name for t in fig.data if t.name is not None]

    assert "Observed Degradation" in trace_names
    assert "GRU Forecaster" in trace_names
    assert "LSTM Forecaster" in trace_names
    assert "Vanilla RNN" in trace_names
    assert "Model E (Adaptive)" in trace_names
