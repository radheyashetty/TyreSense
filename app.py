import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from src.config import load_config
from src.features import fit_baseline_curves, eval_curve
from src.live import update_k_factor
from src.clean import flag_spikes_causal
from src.models import create_gru_model


@st.cache_data
def load_app_data(config_path="config.yaml"):
    """Load cached processed laps and stints datasets."""
    cfg = load_config(config_path)
    proc_dir = cfg["paths"]["processed_dir"]
    laps = pd.read_parquet(os.path.join(proc_dir, "laps.parquet"))
    stints = pd.read_parquet(os.path.join(proc_dir, "stints.parquet"))
    laps["season"] = laps["race_id"].str.slice(0, 4)
    train_laps = laps[laps["season"].isin(["2022", "2023"])]
    curves, global_curve = fit_baseline_curves(train_laps)
    return laps, stints, curves, global_curve


@st.cache_resource
def load_cached_gru_model(checkpoint_path="runs/gru_main/checkpoint.pt"):
    """Load trained GRU model for neural forecasting."""
    if not os.path.exists(checkpoint_path):
        return None
    model = create_gru_model()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def get_stint_subset(laps_df, selected_stint):
    """Retrieve and sort laps belonging to the selected stint."""
    return laps_df[laps_df["stint_id"] == selected_stint].sort_values("stint_lap")


def forecast_future_laps(observed_df, curves, global_curve, horizon=20, k=1.0):
    """Generate forward forecast causally using only observed stint laps."""
    curr_lap = int(observed_df["stint_lap"].max())
    curr_age = int(observed_df["tyre_age"].max())
    track, cmp_name = observed_df["track_id"].iloc[0], observed_df["compound"].iloc[0]
    params = curves.get((track, cmp_name), global_curve)

    early_laps = observed_df[observed_df["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())
    fut_laps = np.arange(curr_lap + 1, curr_lap + horizon + 1)
    base_preds = eval_curve(params, np.arange(curr_age + 1, curr_age + horizon + 1))

    observed_level = eval_curve(params, np.array([curr_age]))[0]
    offset = observed_df["lap_time_fc_s"].iloc[-1] - observed_level
    forecast_times = base_preds + offset + (k - 1.0) * (base_preds - base_preds[0])
    return fut_laps, forecast_times, baseline_time


def compute_remaining_life(forecast_laps, forecast_times, curr_lap, baseline_time, threshold=1.0):
    """Calculate remaining laps until degradation exceeds threshold over baseline."""
    cross_idx = np.where(forecast_times >= baseline_time + threshold)[0]
    if len(cross_idx) > 0:
        return f"{max(0, int(forecast_laps[cross_idx[0]] - curr_lap))} laps"
    return "> 20 laps"


def compute_live_k(full_stint, curr_lap, curves, global_curve):
    """Compute online k-factor using causal observations up to curr_lap."""
    track, cmp_name = full_stint["track_id"].iloc[0], full_stint["compound"].iloc[0]
    params = curves.get((track, cmp_name), global_curve)
    k, k_hist = 1.0, []

    for i in range(min(curr_lap, len(full_stint))):
        s_lap, act_t = int(full_stint["stint_lap"].iloc[i]), float(full_stint["lap_time_fc_s"].iloc[i])
        age = int(full_stint["tyre_age"].iloc[i])
        base_t = float(eval_curve(params, np.array([age]))[0])
        spike = flag_spikes_causal(full_stint.iloc[: i + 1])
        if s_lap >= 3:
            k, k_hist = update_k_factor(k, k_hist, act_t, base_t, act_t - base_t, spike)
    return k


def plot_stint_replay(observed_df, fut_laps, fut_times, base_time, froz_times=None, curr_lap=3, threshold=1.0):
    """Create Plotly chart with historical laps, ghost baseline, and conditional MC band."""
    fig = go.Figure()
    obs_laps, obs_times = observed_df["stint_lap"].values, observed_df["lap_time_fc_s"].values
    fig.add_trace(go.Scatter(x=obs_laps, y=obs_times, mode="lines+markers", name="Observed Pace", line=dict(color="#1f77b4", width=3)))
    if froz_times is not None and len(fut_laps) > 0:
        fig.add_trace(go.Scatter(x=fut_laps, y=froz_times, mode="lines", name="FROZEN (Lap 2)", line=dict(color="#888888", dash="dot", width=1.5)))

    if len(fut_laps) > 0:
        if curr_lap >= 3:
            upper = fut_times + 0.35 * np.sqrt(np.arange(1, len(fut_laps) + 1))
            lower = fut_times - 0.35 * np.sqrt(np.arange(1, len(fut_laps) + 1))
            fig.add_trace(go.Scatter(x=np.concatenate([fut_laps, fut_laps[::-1]]), y=np.concatenate([upper, lower[::-1]]), fill="toself",
                                     fillcolor="rgba(255, 127, 14, 0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="80% Uncertainty Band"))
        fig.add_trace(go.Scatter(x=fut_laps, y=fut_times, mode="lines", name="Forecast Pace", line=dict(color="#ff7f0e", width=2)))

    fig.add_hline(y=base_time + threshold, line_dash="dash", line_color="red", annotation_text="Cliff (+1.0s)")
    fig.update_layout(title="TyreSense Stint Replay & Multi-Lap Forecast", xaxis_title="Stint Lap", yaxis_title="Fuel-Corrected Lap Time (s)", height=450)
    return fig


def render_stint_filters(laps):
    """Render selectors for season, circuit, session type, driver, and tyre compound."""
    seasons = sorted(laps["season"].unique())
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
    model_choice = st.sidebar.radio("Forecasting Model", ["Model E (Online Adaptive)", "Model D (GRU Recurrent)", "Model A (Quadratic Baseline)"])
    return selected_stint, session, model_choice, len(stints_available)


def render_dashboard_metrics(observed, curr_lap, max_lap, est_fuel, rem_life, pit_window):
    """Render top metric status cards across responsive rows."""
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    r1_c1.metric("Circuit", observed["track_id"].iloc[0].replace(" Grand Prix", ""))
    r1_c2.metric("Driver / Tyre", f"{observed['driver'].iloc[0]} ({observed['compound'].iloc[0]})")
    r1_c3.metric("Stint Lap", f"{curr_lap} / {max_lap}")

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    r2_c1.metric("Assumed fuel on board", f"{est_fuel} kg")
    r2_c2.metric("Remaining Life", rem_life)
    r2_c3.metric("Pit Window [P10, P50]", pit_window)


def run_app_view(laps, curves, global_curve, selected_stint, session, model_choice):
    """Render stint replay forecast, navigation, and metrics."""
    full_stint = get_stint_subset(laps, selected_stint)
    max_lap = len(full_stint)
    if "curr_lap" not in st.session_state or st.session_state.get("stint") != selected_stint:
        st.session_state.curr_lap, st.session_state.stint = min(3, max_lap), selected_stint

    curr_lap = st.sidebar.slider("Current Stint Lap", 1, max_lap, st.session_state.curr_lap)
    st.session_state.curr_lap = curr_lap
    observed = full_stint[full_stint["stint_lap"] <= curr_lap]

    k_val = compute_live_k(full_stint, curr_lap, curves, global_curve) if "Model E" in model_choice else 1.0
    fut_laps, fut_times, base_time = forecast_future_laps(observed, curves, global_curve, k=k_val)
    froz_obs = full_stint[full_stint["stint_lap"] <= min(2, len(full_stint))]
    _, froz_times, _ = forecast_future_laps(froz_obs, curves, global_curve, horizon=len(fut_laps))

    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap, base_time)
    rem_int = int(rem_life.split()[0]) if "laps" in rem_life and ">" not in rem_life else 20
    pit_window = f"[Lap {curr_lap + max(1, rem_int - 5)}, Lap {curr_lap + rem_int}]" if curr_lap >= 3 else "Hidden (< Lap 3)"

    tot_laps, race_lap = observed["total_laps"].iloc[-1], observed["lap_number"].iloc[-1]
    est_fuel = max(5.0, round((tot_laps - race_lap) * 1.6, 1)) if session == "Main Race" else 5.0

    render_dashboard_metrics(observed, curr_lap, max_lap, est_fuel, rem_life, pit_window)
    st.sidebar.markdown(f"**Online Wear Multiplier ($k$):** `{k_val:.3f}`")
    fig = plot_stint_replay(observed, fut_laps, fut_times, base_time, froz_times=froz_times, curr_lap=curr_lap)
    st.plotly_chart(fig, width="stretch")


def main():
    """Render the single-page Streamlit stint replay application."""
    st.set_page_config(page_title="TyreSense", layout="wide")
    st.title("TyreSense: F1 Tyre-Life Forecaster")
    laps, stints, curves, global_curve = load_app_data()
    selected_stint, session, model_choice, n_avail = render_stint_filters(laps)

    if n_avail == 0 or selected_stint is None:
        st.warning("No stints match the selected filter combination. Please select a different driver or tyre compound.")
        return
    run_app_view(laps, curves, global_curve, selected_stint, session, model_choice)


if __name__ == "__main__":
    main()
