import numpy as np
from src.features import eval_curve, predict_causal_baseline
from src.live import update_k_factor
from src.clean import flag_spikes_causal


def compute_live_k(full_stint, curr_lap, curves, global_curve):
    """Compute online wear multiplier k using one-step-ahead causal rule."""
    k, k_hist = 1.0, []
    base_preds = predict_causal_baseline(full_stint, curves, global_curve)
    for i in range(1, min(curr_lap, len(full_stint))):
        s_lap = int(full_stint["stint_lap"].iloc[i])
        act_t = float(full_stint["lap_time_fc_s"].iloc[i])
        prev_act = float(full_stint["lap_time_fc_s"].iloc[i - 1])
        delta_b = float(base_preds[i] - base_preds[i - 1])
        spike = flag_spikes_causal(full_stint.iloc[: i + 1])
        if s_lap >= 3:
            k, k_hist = update_k_factor(k, k_hist, act_t, prev_act, delta_b, spike)
    return k


def get_base_curve_predictions(observed_df, curves, global_curve, horizon=20):
    """Evaluate track/compound baseline curve over future tyre ages."""
    curr_age = int(observed_df["tyre_age"].max())
    track, cmp_name = observed_df["track_id"].iloc[0], observed_df["compound"].iloc[0]
    params = curves.get((track, cmp_name), global_curve)
    base_preds = eval_curve(params, np.arange(curr_age + 1, curr_age + horizon + 1))
    observed_level = eval_curve(params, np.array([curr_age]))[0]
    return base_preds, observed_level


def forecast_future_laps(observed_df, curves, global_curve, horizon=20, k=1.0, model_choice="Model E", model_obj=None):
    """Generate multi-step forward degradation and lap time forecast anchored to last observed lap."""
    curr_lap = int(observed_df["stint_lap"].max())
    early_laps = observed_df[observed_df["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())
    fut_laps = np.arange(curr_lap + 1, curr_lap + horizon + 1)
    base_preds, observed_level = get_base_curve_predictions(observed_df, curves, global_curve, horizon)
    last_obs = float(observed_df["lap_time_fc_s"].iloc[-1])

    delta_base = base_preds - observed_level
    if "Model A" in model_choice:
        fut_times = last_obs + delta_base
    elif "Model E" in model_choice or k != 1.0:
        fut_times = last_obs + k * delta_base
    else:
        fut_times = last_obs + delta_base
    return fut_laps, fut_times, baseline_time


def compute_remaining_life(fut_laps, fut_times, curr_lap, baseline_time, threshold=1.0):
    """Calculate remaining laps until degradation exceeds threshold over baseline."""
    deg = fut_times - baseline_time
    cross_idx = np.where(deg >= threshold)[0]
    if len(cross_idx) > 0:
        return f"{max(0, int(fut_laps[cross_idx[0]] - curr_lap))} laps"
    return "> 20 laps"


def compute_cliff_lap_int(fut_laps, fut_times, baseline_time, threshold=1.0):
    """Return predicted absolute cliff stint lap or None."""
    deg = fut_times - baseline_time
    cross_idx = np.where(deg >= threshold)[0]
    if len(cross_idx) > 0:
        return int(fut_laps[cross_idx[0]])
    return None
