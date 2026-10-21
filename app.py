import streamlit as st
from src.app_data import (
    load_app_data, get_stint_subset, load_cached_model,
    load_conformal_quantile, get_data_source_label, get_data_limits_summary
)
from src.app_forecast import (
    forecast_future_laps, compute_live_k,
    compute_remaining_life, compute_cliff_lap_int
)
from src.app_ui import (
    plot_stint_replay, render_sidebar_filters,
    render_metrics_cards, render_data_limits_expander
)


def handle_playback_buttons(max_lap):
    """Render Back, Play/Pause, and Forward navigation controls."""
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("◀ Back"):
        st.session_state.curr_lap = max(1, st.session_state.curr_lap - 1)
    if c2.button("▶ Play"):
        st.session_state.curr_lap = min(max_lap, st.session_state.curr_lap + 1)
    if c3.button("Forward ▶"):
        st.session_state.curr_lap = min(max_lap, st.session_state.curr_lap + 1)


def init_session_state(selected_stint, max_lap):
    """Ensure session state variables for stint replay are initialized."""
    if "curr_lap" not in st.session_state or st.session_state.get("stint") != selected_stint:
        st.session_state.curr_lap = min(3, max_lap)
        st.session_state.stint = selected_stint


def compute_stint_replay_data(full_stint, curr_lap, curves, global_curve, model_choice, model_obj, q_val):
    """Generate observed slice, frozen reference, and multi-step forecast."""
    observed = full_stint[full_stint["stint_lap"] <= curr_lap]
    k_val = compute_live_k(full_stint, curr_lap, curves, global_curve) if "Model E" in model_choice else 1.0

    fut_laps, fut_times, base_time = forecast_future_laps(
        observed, curves, global_curve, k=k_val, model_choice=model_choice, model_obj=model_obj
    )
    froz_obs = full_stint[full_stint["stint_lap"] <= min(2, len(full_stint))]
    _, froz_times, _ = forecast_future_laps(froz_obs, curves, global_curve, horizon=len(fut_laps))

    rem_life = compute_remaining_life(fut_laps, fut_times, curr_lap, base_time)
    froz_cliff = compute_cliff_lap_int(fut_laps, froz_times, base_time)
    roll_cliff = compute_cliff_lap_int(fut_laps, fut_times, base_time)
    return observed, fut_laps, fut_times, froz_times, base_time, rem_life, froz_cliff, roll_cliff, k_val


def render_replay_view(laps, stints, curves, global_curve, season, selected_stint, session, model_choice):
    """Render interactive stint replay chart, metrics cards, and expander."""
    full_stint = get_stint_subset(laps, selected_stint)
    max_lap = len(full_stint)
    init_session_state(selected_stint, max_lap)
    handle_playback_buttons(max_lap)

    curr_lap = st.sidebar.slider("Current Stint Lap", 1, max_lap, st.session_state.curr_lap)
    st.session_state.curr_lap = curr_lap

    model_obj = load_cached_model(model_choice)
    q_val = load_conformal_quantile()
    obs, f_laps, f_times, fr_times, b_time, rem_life, f_cliff, r_cliff, k_val = compute_stint_replay_data(
        full_stint, curr_lap, curves, global_curve, model_choice, model_obj, q_val
    )

    st_row = stints[stints["stint_id"] == selected_stint]
    true_cliff = float(st_row["life_laps"].iloc[0]) if (len(st_row) > 0 and not st_row["censored"].iloc[0]) else None
    tot_laps, race_lap = obs["total_laps"].iloc[-1], obs["lap_number"].iloc[-1]
    est_fuel = max(5.0, round((tot_laps - race_lap) * 1.6, 1)) if session == "Main Race" else 5.0
    d_source = get_data_source_label(season)

    render_metrics_cards(obs, curr_lap, max_lap, est_fuel, rem_life, f_cliff, r_cliff, true_cliff, k_val, d_source)
    fig = plot_stint_replay(obs, f_laps, f_times, b_time, froz_times=fr_times, curr_lap=curr_lap, q_val=q_val)
    st.plotly_chart(fig, width="stretch")
    render_data_limits_expander(d_source, get_data_limits_summary())


def main():
    """Main application loop."""
    st.set_page_config(page_title="TyreSense", layout="wide")
    st.title("TyreSense: F1 Tyre-Life Forecaster")
    laps, stints, curves, global_curve = load_app_data()
    season, selected_stint, model_choice, n_avail = render_sidebar_filters(laps)
    session = "Main Race"

    if n_avail == 0 or selected_stint is None:
        st.warning("No stints match the selected filter combination. Please select a different driver or tyre compound.")
        return
    render_replay_view(laps, stints, curves, global_curve, season, selected_stint, session, model_choice)


if __name__ == "__main__":
    main()
