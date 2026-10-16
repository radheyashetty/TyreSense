import numpy as np
import pandas as pd
from src.fuel import fuel_correct
from src.clean import load_config, clean_race_dataframe, compute_stint_labels


def test_synthetic_fuel_slope_removed_exactly():
    """Verify that fuel correction removes synthetic fuel slope with zero residual error."""
    total_laps = 50
    true_pace = 90.0
    c = 0.03
    lap_numbers = np.arange(1, total_laps + 1)
    synthetic_lap_times = true_pace + c * (total_laps - lap_numbers)
    corrected_times = fuel_correct(synthetic_lap_times, lap_numbers, total_laps, c=c)
    np.testing.assert_allclose(corrected_times, true_pace, atol=1e-12)


def test_attrition_reconciliation():
    """Verify that initial rows minus all dropped filter rows equals clean row count."""
    config = load_config("config.yaml")
    raw_df = pd.read_parquet("data/raw/race_2023_01.parquet")
    clean_laps, _, attrition_df = clean_race_dataframe(raw_df, config)

    initial_rows = int(attrition_df[attrition_df["filter_name"] == "initial_rows"]["rows_remaining"].iloc[0])
    total_dropped = int(attrition_df["rows_dropped"].sum())
    clean_count = len(clean_laps)
    assert initial_rows - total_dropped == clean_count


def test_usable_life_and_censoring():
    """Verify degradation threshold detection and censoring assignment."""
    # Stint with degradation at lap 10
    laps_deg = pd.DataFrame({
        "stint_id": ["stint_1"] * 15,
        "stint_lap": list(range(1, 16)),
        "lap_time_fc_s": [90.0] * 9 + [91.2] + [91.5] * 5
    })
    stints_deg = compute_stint_labels(laps_deg, threshold_s=1.0)
    assert stints_deg["life_laps"].iloc[0] == 10
    assert not stints_deg["censored"].iloc[0]

    # Stint with no degradation (censored)
    laps_censored = pd.DataFrame({
        "stint_id": ["stint_2"] * 15,
        "stint_lap": list(range(1, 16)),
        "lap_time_fc_s": [90.0] * 15
    })
    stints_censored = compute_stint_labels(laps_censored, threshold_s=1.0)
    assert stints_censored["life_laps"].iloc[0] == 15
    assert stints_censored["censored"].iloc[0]
