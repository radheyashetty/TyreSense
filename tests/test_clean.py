import numpy as np
import pandas as pd
from src.fuel import fuel_correct, compute_fuel_coefficient
from src.clean import (
    load_config,
    clean_race_dataframe,
    compute_stint_labels,
    flag_spikes_offline,
    flag_spikes_causal,
)


def test_synthetic_fuel_slope_removed_exactly():
    """Verify that fuel correction removes synthetic fuel slope with zero residual error."""
    total_laps = 50
    true_pace = 90.0
    c = 0.03
    lap_numbers = np.arange(1, total_laps + 1)
    synthetic_lap_times = true_pace + c * (total_laps - lap_numbers)
    corrected_times = fuel_correct(synthetic_lap_times, lap_numbers, total_laps, c=c)
    np.testing.assert_allclose(corrected_times, true_pace, atol=1e-12)


def test_track_scaled_fuel_mode():
    """Verify track-scaled fuel coefficient computation and exact slope removal."""
    cfg = {
        "fuel": {"mode": "track_scaled", "base_c": 0.03, "ref_km": 5.0},
        "circuit_length_km": {"Bahrain Grand Prix": 5.412}
    }
    c_scaled = compute_fuel_coefficient("Bahrain Grand Prix", cfg)
    expected_c = 0.03 * (5.412 / 5.0)
    assert abs(c_scaled - expected_c) < 1e-9

    total_laps = 57
    true_pace = 95.0
    lap_numbers = np.arange(1, total_laps + 1)
    synthetic = true_pace + c_scaled * (total_laps - lap_numbers)
    corrected = fuel_correct(synthetic, lap_numbers, total_laps, c=c_scaled)
    np.testing.assert_allclose(corrected, true_pace, atol=1e-12)


def test_spike_filter_offline_and_causal():
    """Verify that an isolated spike with 1-lap recovery is flagged offline and causally."""
    times = [90.0, 90.1, 90.0, 90.2, 92.8, 90.1, 90.2, 90.3]
    stint = pd.DataFrame({"lap_time_fc_s": times})
    offline_flags = flag_spikes_offline(stint, threshold_s=2.0, recover_s=1.0, window=3)
    assert bool(offline_flags.iloc[4]) is True
    assert bool(offline_flags.iloc[3]) is False

    history = stint.iloc[:5]
    causal_flag = flag_spikes_causal(history, threshold_s=2.0, window=3)
    assert causal_flag is True


def test_slow_lap_without_recovery_not_flagged_offline():
    """Verify that permanent pace degradation (no recovery) is not flagged as a spike."""
    times = [90.0, 90.1, 90.0, 90.2, 92.8, 92.9, 93.1, 93.2]
    stint = pd.DataFrame({"lap_time_fc_s": times})
    offline_flags = flag_spikes_offline(stint, threshold_s=2.0, recover_s=1.0, window=3)
    assert not offline_flags.any(), "Permanent degradation was mistakenly flagged as a spike"


def test_attrition_reconciliation():
    """Verify that initial rows minus all dropped filter rows equals clean row count."""
    config = load_config("config.yaml")
    raw_df = pd.read_parquet("data/raw/race_2023_01.parquet")
    clean_laps, _, attrition_df = clean_race_dataframe(raw_df, config)

    initial_rows = int(attrition_df[attrition_df["filter_name"] == "initial_rows"]["rows_remaining"].iloc[0])
    drop_rows = int(attrition_df[attrition_df["filter_name"].str.startswith("drop_")]["rows_dropped"].sum())
    clean_count = len(clean_laps)
    assert initial_rows - drop_rows == clean_count


def test_usable_life_and_censoring():
    """Verify degradation threshold detection and censoring assignment."""
    laps_deg = pd.DataFrame({
        "stint_id": ["stint_1"] * 15,
        "stint_lap": list(range(1, 16)),
        "lap_time_fc_s": [90.0] * 9 + [91.2] + [91.5] * 5
    })
    stints_deg = compute_stint_labels(laps_deg, threshold_s=1.0)
    assert stints_deg["life_laps"].iloc[0] == 10
    assert not stints_deg["censored"].iloc[0]

    laps_censored = pd.DataFrame({
        "stint_id": ["stint_2"] * 15,
        "stint_lap": list(range(1, 16)),
        "lap_time_fc_s": [90.0] * 15
    })
    stints_censored = compute_stint_labels(laps_censored, threshold_s=1.0)
    assert stints_censored["life_laps"].iloc[0] == 15
    assert stints_censored["censored"].iloc[0]
