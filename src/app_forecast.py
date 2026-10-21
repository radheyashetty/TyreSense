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


def predict_neural_residuals(model_obj, observed_df, track_map, compound_map, curves, global_curve):
    """Run neural model forward pass on observed laps to predict future residuals."""
    import torch
    from src.features import build_stint_features
    feats = build_stint_features(observed_df, track_map, compound_map, include_temp_delta=True)
    baseline = predict_causal_baseline(observed_df, curves, global_curve)
    curr_res = (observed_df["lap_time_fc_s"].values - baseline).reshape(-1, 1)
    num_cols = ["tyre_age", "lap_fraction", "relative_pace", "track_temp", "track_status", "gap_mask", "track_temp_delta_c"]
    full_numeric = np.hstack([feats[num_cols].values, curr_res])

    seq_len = 10
    x_pad = np.zeros((seq_len, 8), dtype=np.float32)
    trk_pad = np.zeros(seq_len, dtype=np.int64)
    cmp_pad = np.zeros(seq_len, dtype=np.int64)

    obs_len = min(len(observed_df), seq_len)
    x_pad[-obs_len:] = full_numeric[-obs_len:]
    trk_pad[-obs_len:] = feats["track"].values[-obs_len:]
    cmp_pad[-obs_len:] = feats["compound"].values[-obs_len:]

    x_tensor = torch.tensor(x_pad).unsqueeze(0).float()
    trk_tensor = torch.tensor(trk_pad).unsqueeze(0).long()
    cmp_tensor = torch.tensor(cmp_pad).unsqueeze(0).long()
    model_obj.eval()
    with torch.no_grad():
        res = model_obj(x_tensor, trk_tensor, cmp_tensor).cpu().numpy()[0]
    return res


def forecast_future_laps(observed_df, curves, global_curve, horizon=20, k=1.0, model_choice="Model E", model_obj=None, track_map=None, compound_map=None, mode="continuity_adjusted"):
    """Generate forecast under continuity-adjusted UI mode or direct target reconstruction."""
    curr_lap = int(observed_df["stint_lap"].max())
    early_laps = observed_df[observed_df["stint_lap"] <= 3]
    baseline_time = float(early_laps["lap_time_fc_s"].mean())
    fut_laps = np.arange(curr_lap + 1, curr_lap + horizon + 1)
    base_preds, observed_level = get_base_curve_predictions(observed_df, curves, global_curve, horizon)
    last_obs = float(observed_df["lap_time_fc_s"].iloc[-1])
    delta_base = base_preds - observed_level

    is_neural = model_obj is not None and any(m in model_choice for m in ["GRU", "LSTM", "RNN"])
    if is_neural:
        if track_map is None or compound_map is None:
            from src.app_data import get_cached_maps
            track_map, compound_map = get_cached_maps()
        res = predict_neural_residuals(model_obj, observed_df, track_map, compound_map, curves, global_curve)
        if mode == "direct_reconstruction":
            base_obs = float(predict_causal_baseline(observed_df, curves, global_curve)[-1])
            fut_times = last_obs + delta_base + (res[:horizon] - (last_obs - base_obs))
        else:
            fut_times = last_obs + delta_base + (res[:horizon] - res[0])
    elif "Model A" in model_choice:
        fut_times = last_obs + delta_base
    elif "Model E" in model_choice or k != 1.0:
        fut_times = last_obs + k * delta_base
    else:
        fut_times = last_obs + delta_base
    return fut_laps, fut_times, baseline_time


def compute_cliff_lap_int(fut_laps, fut_times, baseline_time, threshold=1.0):
    """Return first lap of 2 consecutive laps where degradation >= threshold, or None."""
    deg = fut_times - baseline_time
    for i in range(len(deg) - 1):
        if deg[i] >= threshold and deg[i + 1] >= threshold:
            return int(fut_laps[i])
    return None


def compute_remaining_life(fut_laps, fut_times, curr_lap, baseline_time, threshold=1.0):
    """Calculate remaining laps until degradation satisfies 2-consecutive-lap cliff rule."""
    cliff_lap = compute_cliff_lap_int(fut_laps, fut_times, baseline_time, threshold)
    if cliff_lap is not None:
        return f"{max(0, int(cliff_lap - curr_lap))} laps"
    return "> 20 laps"


def compute_all_model_forecasts(observed_df, curves, global_curve, k_val, horizon=20):
    """Compute forecasts, cliff laps, and remaining lives for all 4 models on the same stint."""
    from src.app_data import load_cached_model, get_cached_maps
    track_map, compound_map = get_cached_maps()
    models_spec = [
        ("GRU Forecaster", "GRU Forecaster (Primary Recurrent Net)", load_cached_model("GRU"), "#ff7f0e", "solid"),
        ("LSTM Forecaster", "LSTM Forecaster (Cell-State Memory)", load_cached_model("LSTM"), "#2ca02c", "dash"),
        ("Vanilla RNN", "Vanilla RNN (Baseline Recurrent Net)", load_cached_model("RNN"), "#9467bd", "dot"),
        ("Model E (Adaptive)", "Neural-Guided Adaptive (Model E)", None, "#d62728", "longdash"),
    ]
    curr_lap = int(observed_df["stint_lap"].max())
    results = []
    for short_name, full_name, model_obj, color, dash in models_spec:
        f_laps, f_times, b_time = forecast_future_laps(
            observed_df, curves, global_curve, horizon=horizon, k=k_val,
            model_choice=full_name, model_obj=model_obj, track_map=track_map, compound_map=compound_map
        )
        cliff = compute_cliff_lap_int(f_laps, f_times, b_time)
        life = compute_remaining_life(f_laps, f_times, curr_lap, b_time)
        results.append({
            "name": short_name, "full_name": full_name, "fut_laps": f_laps,
            "fut_times": f_times, "base_time": b_time, "cliff_lap": cliff,
            "rem_life": life, "color": color, "dash": dash
        })
    return results
