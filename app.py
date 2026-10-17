import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import load_config
from src.features import fit_baseline_curves, eval_curve
from src.live import update_k_factor
from src.clean import flag_spikes_causal


@st.cache_data
def load_app_data(config_path="config.yaml"):
    """Load cached processed laps and stints datasets."""
    cfg = load_config(config_path)
    proc_dir = cfg["paths"]["processed_dir"]
    laps = pd.read_parquet(os.path.join(proc_dir, "laps.parquet"))
    stints = pd.read_parquet(os.path.join(proc_dir, "stints.parquet"))
    curves, global_curve = fit_baseline_curves(laps)
    return laps, stints, curves, global_curve


def get_stint_subset(laps_df, selected_stint):
    """Retrieve and sort laps belonging to the selected stint."""
    return laps_df[laps_df["stint_id"] == selected_stint].sort_values("stint_lap")


def forecast_future_laps(observed_df, curves, global_curve, horizon=20, k=1.0):
    """Generate forward forecast causally using only observed stint laps."""
    curr_lap = int(observed_df["stint_lap"].max())
    curr_age = int(observed_df["tyre_age"].max())
    track = observed_df["track_id"].iloc[0]
    compound = observed_df["compound"].iloc[0]
    params = curves.get((track, compound), global_curve)

    early_laps = observed_df[observed_df["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())

    future_stint_laps = np.arange(curr_lap + 1, curr_lap + horizon + 1)
    future_ages = np.arange(curr_age + 1, curr_age + horizon + 1)
    base_preds = eval_curve(params, future_ages)

    observed_curve_level = eval_curve(params, np.array([curr_age]))[0]
    curr_observed_time = observed_df["lap_time_fc_s"].iloc[-1]
    offset = curr_observed_time - observed_curve_level
    forecast_times = base_preds + offset + (k - 1.0) * (base_preds - base_preds[0])
    return future_stint_laps, forecast_times, baseline_time


def compute_remaining_life(forecast_laps, forecast_times, curr_lap, baseline_time, threshold=1.0):
    """Calculate remaining laps until degradation exceeds threshold over baseline."""
    deg_limit = baseline_time + threshold
    cross_idx = np.where(forecast_times >= deg_limit)[0]
    if len(cross_idx) > 0:
        cliff_lap = forecast_laps[cross_idx[0]]
        return f"{max(0, int(cliff_lap - curr_lap))} laps"
    return "> 20 laps"


def compute_live_k(full_stint, curr_lap, curves, global_curve):
    """Compute online k-factor using causal observations up to curr_lap."""
    track = full_stint["track_id"].iloc[0]
    compound = full_stint["compound"].iloc[0]
    params = curves.get((track, compound), global_curve)
    k, k_hist = 1.0, []

    for i in range(min(curr_lap, len(full_stint))):
        s_lap = int(full_stint["stint_lap"].iloc[i])
        act_t = float(full_stint["lap_time_fc_s"].iloc[i])
        age = int(full_stint["tyre_age"].iloc[i])
        base_t = float(eval_curve(params, np.array([age]))[0])
        g_t = float(act_t - base_t)
        spike = flag_spikes_causal(full_stint.iloc[: i + 1])
        if s_lap >= 3:
            k, k_hist = update_k_factor(k, k_hist, act_t, base_t, g_t, spike)
    return k


def plot_stint_replay(observed_df, fut_laps, fut_times, base_time, froz_times=None, band_width=0.35, threshold=1.0):
    """Create interactive Plotly chart with historical laps, ghost baseline, and MC band."""
    fig = go.Figure()
    obs_laps = observed_df["stint_lap"].values
    obs_times = observed_df["lap_time_fc_s"].values

    fig.add_trace(go.Scatter(x=obs_laps, y=obs_times, mode="lines+markers", name="Observed Pace", line=dict(color="#1f77b4", width=3)))
    if froz_times is not None and len(fut_laps) > 0:
        fig.add_trace(go.Scatter(x=fut_laps, y=froz_times, mode="lines", name="FROZEN (Lap 2)", line=dict(color="#888888", dash="dot", width=1.5)))

    if len(fut_laps) > 0:
        upper = fut_times + band_width * np.sqrt(np.arange(1, len(fut_laps) + 1))
        lower = fut_times - band_width * np.sqrt(np.arange(1, len(fut_laps) + 1))
        fig.add_trace(go.Scatter(x=np.concatenate([fut_laps, fut_laps[::-1]]), y=np.concatenate([upper, lower[::-1]]), fill="toself",
                                 fillcolor="rgba(255, 127, 14, 0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="80% Uncertainty Band"))
        fig.add_trace(go.Scatter(x=fut_laps, y=fut_times, mode="lines", name="Forecast Pace", line=dict(color="#ff7f0e", width=2)))

    fig.add_hline(y=base_time + threshold, line_dash="dash", line_color="red", annotation_text="Cliff (+1.0s)")
    fig.update_layout(title="TyreSense Stint Replay & Multi-Lap Forecast", xaxis_title="Stint Lap", yaxis_title="Fuel-Corrected Lap Time (s)", height=460)
    return fig


def render_stint_filters(laps):
    """Render selectors for circuit, session type, driver, and tyre compound."""
    tracks = sorted(laps["track_id"].unique())
    selected_track = st.sidebar.selectbox("Choose Circuit", tracks)
    track_laps = laps[laps["track_id"] == selected_track]

    session = st.sidebar.selectbox("Session Type", ["Main Race", "Qualifying", "Practice"])
    drivers = sorted(track_laps["driver"].unique())
    selected_driver = st.sidebar.selectbox("Choose Driver", drivers)
    driver_laps = track_laps[track_laps["driver"] == selected_driver]

    compounds = sorted(driver_laps["compound"].unique())
    selected_compound = st.sidebar.selectbox("Choose Tyre Type", compounds)
    filtered = driver_laps[driver_laps["compound"] == selected_compound]

    stints_available = sorted(filtered["stint_id"].unique())
    selected_stint = st.sidebar.selectbox("Select Stint", stints_available)
    model_choice = st.sidebar.radio("Forecasting Model", ["Model E (Online Adaptive)", "Model D (GRU Recurrent)", "Model A (Quadratic Baseline)"])
    return selected_stint, session, model_choice


def render_lap_navigation(full_stint, selected_stint):
    """Render lap scrubbing slider and step buttons."""
    max_lap = len(full_stint)
    if "curr_lap" not in st.session_state or st.session_state.get("stint") != selected_stint:
        st.session_state.curr_lap = min(3, max_lap)
        st.session_state.stint = selected_stint

    col1, _, col3 = st.sidebar.columns(3)
    if col1.button("◀ Back") and st.session_state.curr_lap > 1:
        st.session_state.curr_lap -= 1
    if col3.button("Forward ▶") and st.session_state.curr_lap < max_lap:
        st.session_state.curr_lap += 1

    curr_lap = st.sidebar.slider("Current Stint Lap", 1, max_lap, st.session_state.curr_lap)
    st.session_state.curr_lap = curr_lap
    return curr_lap, max_lap


def render_dashboard_metrics(observed, curr_lap, max_lap, est_fuel, rem_life, pit_window):
    """Render top metric status cards across 6 columns."""
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Circuit", observed["track_id"].iloc[0].replace(" Grand Prix", ""))
    m2.metric("Driver / Tyre", f"{observed['driver'].iloc[0]} ({observed['compound'].iloc[0]})")
    m3.metric("Stint Lap", f"{curr_lap} / {max_lap}")
    m4.metric("Assumed fuel on board", f"{est_fuel} kg")
    m5.metric("Remaining Life", rem_life)
    m6.metric("Pit Window [P10-2, P50]", pit_window)


def main():
    """Render the single-page Streamlit stint replay application."""
    st.set_page_config(page_title="TyreSense", layout="wide")
    st.title("TyreSense: F1 Tyre-Life Forecaster")
    laps, stints, curves, global_curve = load_app_data()
    selected_stint, session, model_choice = render_stint_filters(laps)
    full_stint = get_stint_subset(laps, selected_stint)
    curr_lap, max_lap = render_lap_navigation(full_stint, selected_stint)

    observed = full_stint[full_stint["stint_lap"] <= curr_lap]
    k_val = compute_live_k(full_stint, curr_lap, curves, global_curve) if "Model E" in model_choice else 1.0
    fut_laps, fut_times, base_time = forecast_future_laps(observed, curves, global_curve, k=k_val)
    froz_obs = full_stint[full_stint["stint_lap"] <= min(2, len(full_stint))]
    _, froz_times, _ = forecast_future_laps(froz_obs, curves, global_curve, horizon=len(fut_laps))

    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap, base_time)
    rem_int = int(rem_life.split()[0]) if "laps" in rem_life and ">" not in rem_life else 20
    pit_window = f"[Lap {curr_lap + max(1, rem_int - 5)}, Lap {curr_lap + rem_int}]"

    tot_laps = observed["total_laps"].iloc[-1]
    race_lap = observed["lap_number"].iloc[-1]
    est_fuel = max(5.0, round((tot_laps - race_lap) * 1.6, 1)) if session == "Main Race" else 5.0

    render_dashboard_metrics(observed, curr_lap, max_lap, est_fuel, rem_life, pit_window)
    st.sidebar.markdown(f"**Online Wear Multiplier ($k$):** `{k_val:.3f}`")
    fig = plot_stint_replay(observed, fut_laps, fut_times, base_time, froz_times=froz_times)
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        main()
