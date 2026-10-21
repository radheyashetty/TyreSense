import random
import pytest
import pandas as pd
from src.openf1_convert import convert_openf1_bundle
from tests.data.invariant_helpers import (
    check_schema_and_dtypes, check_duplicates, check_lap_start_at_one,
    check_contiguous_laps, check_compounds_valid, check_tyre_life_rules
)


def generate_driver_laps_and_stints(d, d_laps, compounds, rng, base_time):
    """Generate mock stints and laps for a single driver."""
    stints, laps = [], []
    n_stints = rng.randint(1, 3)
    stint_len = max(1, d_laps // n_stints)
    curr_lap, cum_sec = 1, 0.0
    for s_idx in range(1, n_stints + 1):
        s_end = min(d_laps, curr_lap + stint_len - 1) if s_idx < n_stints else d_laps
        c_choice = compounds[(d + s_idx) % len(compounds)]
        stints.append({
            "driver_number": d, "stint_number": s_idx, "compound": c_choice,
            "lap_start": curr_lap, "lap_end": s_end, "tyre_age_at_start": 0
        })
        for ln in range(curr_lap, s_end + 1):
            dur = 85.0 + rng.random() * 5.0
            cum_sec += dur
            is_out = (ln == 1 or ln == curr_lap)
            laps.append({
                "driver_number": d, "lap_number": ln, "lap_duration": dur,
                "date_start": (base_time + pd.Timedelta(seconds=cum_sec)).isoformat(),
                "duration_sector_1": 28.0, "duration_sector_2": 28.0, "duration_sector_3": dur - 56.0,
                "i1_speed": 290, "i2_speed": 280, "st_speed": 310, "is_pit_out_lap": is_out
            })
        curr_lap = s_end + 1
        if curr_lap > d_laps:
            break
    return stints, laps


def generate_random_openf1_bundle(seed, n_drivers, n_race_laps):
    """Generate a valid random OpenF1 data bundle using a fixed seed."""
    rng = random.Random(seed)
    compounds = ["SOFT", "MEDIUM", "HARD"]
    drivers = [{"driver_number": i + 1, "name_acronym": f"D{i+1:02d}"} for i in range(n_drivers)]
    stints, laps = [], []
    base_time = pd.Timestamp("2023-05-10 14:00:00+00:00")
    for d_info in drivers:
        d = d_info["driver_number"]
        is_retired = (rng.random() < 0.15)
        d_laps = rng.randint(15, n_race_laps) if is_retired else n_race_laps
        d_stints, d_laps_list = generate_driver_laps_and_stints(d, d_laps, compounds, rng, base_time)
        stints.extend(d_stints)
        laps.extend(d_laps_list)
    weather = [{"date": base_time.isoformat(), "track_temperature": 35.0, "air_temperature": 25.0, "humidity": 50.0, "rainfall": False}]
    return {"drivers": drivers, "stints": stints, "laps": laps, "weather": weather, "race_control": []}


@pytest.mark.slow
def test_200_random_property_races():
    """Verify invariants hold across 200 random synthetic races with fixed seeds."""
    master_seed = 42
    rng = random.Random(master_seed)
    for i in range(200):
        seed = rng.randint(1, 1000000)
        n_drivers = rng.randint(18, 22)
        n_laps = rng.randint(45, 65)
        bundle = generate_random_openf1_bundle(seed, n_drivers, n_laps)
        df = convert_openf1_bundle(bundle, {"event_name": f"Synthetic_{i}"}, 1, 2023)
        assert check_schema_and_dtypes(df), f"Race {i}: schema error"
        assert check_duplicates(df), f"Race {i}: duplicates found"
        assert check_lap_start_at_one(df)[0], f"Race {i}: lap 1 start error"
        assert check_contiguous_laps(df)[0], f"Race {i}: contiguous laps error"
        assert check_compounds_valid(df)[0], f"Race {i}: invalid compounds"
        assert check_tyre_life_rules(df)[0], f"Race {i}: tyre life rule error"
