import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import load_config
from src.features import fit_baseline_curves, eval_curve


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


def forecast_future_laps(observed_df, curves, global_curve, horizon=20):
    """Generate 20-lap forward forecast causally using only observed stint laps."""
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

    # Offset curve so it starts from the observed pace level
    observed_curve_level = eval_curve(params, np.array([curr_age]))[0]
    curr_observed_time = observed_df["lap_time_fc_s"].iloc[-1]
    offset = curr_observed_time - observed_curve_level
    forecast_times = base_preds + offset

    return future_stint_laps, forecast_times, baseline_time


def compute_remaining_life(forecast_laps, forecast_times, curr_lap, baseline_time, threshold=1.0):
    """Calculate remaining laps until degradation exceeds 1.0 s over baseline."""
    deg_limit = baseline_time + threshold
    cross_idx = np.where(forecast_times >= deg_limit)[0]
    if len(cross_idx) > 0:
        cliff_lap = forecast_laps[cross_idx[0]]
        return f"{max(0, cliff_lap - curr_lap)} laps"
    return "> 20 laps"


def plot_stint_replay(observed_df, fut_laps, fut_times, baseline_time, threshold=1.0):
    """Create interactive Plotly chart with historical laps and forward forecast."""
    fig = go.Figure()
    obs_laps = observed_df["stint_lap"].values
    obs_times = observed_df["lap_time_fc_s"].values

    fig.add_trace(go.Scatter(x=obs_laps, y=obs_times, mode="lines+markers", name="Observed Pace", line=dict(color="#1f77b4", width=3)))
    if len(fut_laps) > 0:
        fig.add_trace(go.Scatter(x=fut_laps, y=fut_times, mode="lines", name="20-Lap Forecast", line=dict(color="#ff7f0e", dash="dash", width=2)))

    deg_line = baseline_time + threshold
    fig.add_hline(y=deg_line, line_dash="dot", line_color="red", annotation_text="Cliff (+1.0s)")
    fig.update_layout(title="TyreSense Stint Replay & Forecast", xaxis_title="Stint Lap", yaxis_title="Fuel-Corrected Lap Time (s)", height=450)
    return fig


def render_stint_filters(laps):
    """Render selectors for circuit, session type, driver, and tyre compound."""
    tracks = sorted(laps["track_id"].unique())
    selected_track = st.sidebar.selectbox("Choose Circuit", tracks)
    track_laps = laps[laps["track_id"] == selected_track]

    session = st.sidebar.selectbox("Session Type", ["Main Race", "Qualifying", "Practice"])
    if session == "Qualifying":
        st.sidebar.warning("Qualifying runs 1 flying lap on ~5kg fuel. Showing Race stint telemetry.")
    elif session == "Practice":
        st.sidebar.info("Practice long runs simulate race stints. Showing continuous stint telemetry.")

    drivers = sorted(track_laps["driver"].unique())
    selected_driver = st.sidebar.selectbox("Choose Driver", drivers)
    driver_laps = track_laps[track_laps["driver"] == selected_driver]

    compounds = sorted(driver_laps["compound"].unique())
    selected_compound = st.sidebar.selectbox("Choose Tyre Type", compounds)
    filtered = driver_laps[driver_laps["compound"] == selected_compound]

    stints_available = sorted(filtered["stint_id"].unique())
    selected_stint = st.sidebar.selectbox("Select Stint", stints_available)
    return selected_stint, session


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


def main():
    """Render the single-page Streamlit stint replay application."""
    st.set_page_config(page_title="TyreSense", layout="wide")
    st.title("TyreSense: F1 Tyre-Life Forecaster")
    laps, stints, curves, global_curve = load_app_data()
    selected_stint, session = render_stint_filters(laps)
    full_stint = get_stint_subset(laps, selected_stint)
    curr_lap, max_lap = render_lap_navigation(full_stint, selected_stint)
    observed = full_stint[full_stint["stint_lap"] <= curr_lap]
    fut_laps, fut_times, base_time = forecast_future_laps(observed, curves, global_curve)
    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap, base_time)

    tot_laps = observed["total_laps"].iloc[-1]
    race_lap = observed["lap_number"].iloc[-1]
    est_fuel = max(5.0, round((tot_laps - race_lap) * 1.6, 1)) if session == "Main Race" else 5.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Circuit", observed["track_id"].iloc[0].replace(" Grand Prix", ""))
    m2.metric("Driver / Tyre", f"{observed['driver'].iloc[0]} ({observed['compound'].iloc[0]})")
    m3.metric("Stint Lap", f"{curr_lap} / {max_lap}")
    m4.metric("Est. Fuel", f"{est_fuel} kg")
    m5.metric("Remaining Life", rem_life)

    fig = plot_stint_replay(observed, fut_laps, fut_times, base_time)
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    else:
        main()
