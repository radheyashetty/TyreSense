import numpy as np
import pandas as pd
from src.clean import flag_spikes_causal


def update_k_factor(k, k_history, actual_t, base_t, g_t, is_spike=False, eta=0.3, k_min=0.5, k_max=2.0, min_g=0.05, smooth_laps=3):
    """Update tyre wear multiplier k using normalized error-correction rule."""
    if is_spike or abs(g_t) <= min_g:
        return k, k_history
    pred = base_t + k * g_t
    err = actual_t - pred
    k_raw = k + eta * g_t * err / (g_t ** 2 + 0.05)
    k_history.append(float(k_raw))
    window = k_history[-smooth_laps:]
    k_new = float(np.clip(np.mean(window), k_min, k_max))
    return k_new, k_history


def compute_cliff_life(forecast_deg, threshold_s=1.0):
    """Determine remaining laps until forecast degradation crosses threshold."""
    cross_idx = np.where(forecast_deg >= threshold_s)[0]
    if len(cross_idx) > 0:
        return int(cross_idx[0] + 1)
    return None


def run_live_correction_stint(stint_df, base_times, g_preds, threshold_s=1.0, eta=0.3):
    """Simulate online stint with normalized error-correction updating multiplier k."""
    k = 1.0
    k_history = []
    records = []
    num_laps = len(stint_df)

    for idx in range(num_laps):
        stint_lap = int(stint_df["stint_lap"].iloc[idx])
        actual_t = float(stint_df["lap_time_fc_s"].iloc[idx])
        base_t = float(base_times[idx])
        g_t = float(g_preds[idx, 0]) if g_preds.ndim == 2 else float(g_preds[idx])

        history_df = stint_df.iloc[: idx + 1]
        is_spike = flag_spikes_causal(history_df)

        if stint_lap >= 3:
            k, k_history = update_k_factor(k, k_history, actual_t, base_t, g_t, is_spike, eta)

        fut_deg = k * g_preds[idx]
        life = compute_cliff_life(fut_deg, threshold_s)
        records.append({"stint_lap": stint_lap, "k": k, "is_spike": is_spike, "remaining_life": life})

    return pd.DataFrame(records)
