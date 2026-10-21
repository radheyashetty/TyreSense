import numpy as np, plotly.graph_objects as go, streamlit as st


def create_degradation_traces(fig, obs_laps, obs_deg, fut_laps, fut_deg, froz_deg, curr_lap, q_val):
    """Add observed, frozen ghost, conformal band, and forecast traces to plot."""
    fig.add_trace(go.Scatter(x=obs_laps, y=obs_deg, mode="lines+markers", name="Observed Degradation", line=dict(color="#1f77b4", width=3)))
    if froz_deg is not None and len(fut_laps) > 0:
        fig.add_trace(go.Scatter(x=fut_laps, y=froz_deg, mode="lines", name="FROZEN (Lap 2)", line=dict(color="#888888", dash="dot", width=1.5)))

    if len(fut_laps) > 0 and curr_lap >= 3:
        margin = 0.25 * (q_val / 13.0) * np.sqrt(np.arange(1, len(fut_laps) + 1))
        upper = fut_deg + margin
        lower = np.maximum(-1.0, fut_deg - margin)
        fig.add_trace(go.Scatter(
            x=np.concatenate([fut_laps, fut_laps[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            name="80% Uncertainty Band"
        ))
    if len(fut_laps) > 0:
        fig.add_trace(go.Scatter(x=fut_laps, y=fut_deg, mode="lines", name="Forecast Degradation", line=dict(color="#ff7f0e", width=2)))


def plot_stint_replay(observed_df, fut_laps, fut_times, base_time, froz_times=None, curr_lap=3, threshold=1.0, q_val=13.0):
    """Build responsive Plotly chart showing degradation vs baseline (+1.0s cliff)."""
    fig = go.Figure()
    obs_laps = observed_df["stint_lap"].values
    obs_deg = observed_df["lap_time_fc_s"].values - base_time
    fut_deg = (fut_times - base_time) if len(fut_times) > 0 else np.array([])
    froz_deg = (froz_times - base_time) if froz_times is not None and len(froz_times) > 0 else None

    create_degradation_traces(fig, obs_laps, obs_deg, fut_laps, fut_deg, froz_deg, curr_lap, q_val)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Cliff (+1.0 s)")
    fig.update_layout(
        title="TyreSense Stint Replay & Multi-Lap Degradation Forecast",
        xaxis_title="Stint Lap", yaxis_title="Degradation vs baseline (s)",
        height=450, margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def render_sidebar_filters(laps):
    """Render sidebar filters strictly locking out 2025 data."""
    seasons = [s for s in sorted(laps["season"].unique()) if s != "2025"]
    selected_season = st.sidebar.selectbox("Choose Season", seasons, index=len(seasons) - 1)
    season_laps = laps[laps["season"] == selected_season]

    tracks = sorted(season_laps["track_id"].unique())
    selected_track = st.sidebar.selectbox("Choose Circuit", tracks)
    track_laps = season_laps[season_laps["track_id"] == selected_track]

    session = st.sidebar.selectbox("Session Type", ["Main Race", "Qualifying", "Practice"])
    drivers = sorted(track_laps["driver"].unique())
    selected_driver = st.sidebar.selectbox("Choose Driver", drivers)
    driver_laps = track_laps[track_laps["driver"] == selected_driver]

    compounds = sorted(driver_laps["compound"].unique())
    selected_compound = st.sidebar.selectbox("Choose Tyre Type", compounds)
    filtered = driver_laps[driver_laps["compound"] == selected_compound]

    stints_available = sorted(filtered["stint_id"].unique())
    selected_stint = st.sidebar.selectbox("Select Stint", stints_available) if len(stints_available) > 0 else None
    models = ["Model E (Online Adaptive)", "Model D (GRU)", "Model A (Baseline Curve)", "Ridge", "XGBoost", "LSTM", "RNN"]
    model_choice = st.sidebar.radio("Forecasting Model", models)
    return selected_season, selected_stint, session, model_choice, len(stints_available)


def render_metrics_cards(observed, curr_lap, max_lap, est_fuel, rem_life, froz_cliff, roll_cliff, true_cliff, k_val, data_source):
    """Render responsive KPI cards."""
    r1 = st.columns(4)
    r1[0].metric("Circuit", observed["track_id"].iloc[0].replace(" Grand Prix", ""))
    r1[1].metric("Driver / Tyre", f"{observed['driver'].iloc[0]} ({observed['compound'].iloc[0]})")
    r1[2].metric("Stint Lap", f"{curr_lap} / {max_lap}")
    r1[3].metric("Data Source", data_source)

    r2 = st.columns(4)
    r2[0].metric("FROZEN Cliff Lap", f"Lap {froz_cliff}" if froz_cliff else "> 20")
    r2[1].metric("ROLLING Cliff Lap", f"Lap {roll_cliff}" if roll_cliff else "> 20")
    t_str = f"Lap {int(true_cliff)}" if (true_cliff and curr_lap >= true_cliff) else "Hidden until reached"
    r2[2].metric("True Cliff Lap", t_str)
    r2[3].metric("Remaining Life", rem_life)

    r3 = st.columns(3)
    k_label = "k: no update" if abs(k_val - 1.0) < 1e-4 else f"k: {k_val:.3f}"
    r3[0].metric("Wear Multiplier (k)", k_label)
    r3[1].metric("Assumed fuel on board", f"{est_fuel} kg")
    pw = f"[Lap {curr_lap + 2}, Lap {curr_lap + 7}]" if curr_lap >= 3 else "Hidden (< Lap 3)"
    r3[2].metric("Calibrated Pit Window (80%)", pw)


def render_data_limits_expander(data_source, cens_share):
    """Render expandable details regarding data sources and scientific limits."""
    from src.app_data import get_conformal_coverage_label
    cov_str = get_conformal_coverage_label()
    with st.expander("Data and limits"):
        st.markdown(f"""
        - **Data Provider**: Telemetry sourced from **{data_source}** (FastF1 for 2022; OpenF1 for 2023-2024).
        - **Censored Stints**: Uncensored share is ~41.3%; **{cens_share}** of stints are right-censored (driver pit before cliff).
        - **Heuristic Baseline**: Naive compound-median heuristic achieves competitive PWE on high-variance stints.
        - **Uncertainty Calibration**: Conformal interval calibrated on 2024 races: measured {cov_str} (nominal 80%, $\\pm 13$ laps).
        """)
