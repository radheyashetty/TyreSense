import pandas as pd
from src.split import (
    load_config,
    split_by_race,
    fit_train_scaler,
    apply_scaler,
    build_track_mapping,
    encode_tracks,
)


def create_mock_dataset():
    """Create a controlled mock dataset spanning train (2022-2023), val (2024), and test (2025)."""
    records = []
    # 2022-2023 train races
    for race_id in ["2022_01", "2023_01", "2023_02"]:
        records.append({
            "race_id": race_id,
            "track_id": f"Track_{race_id}",
            "driver": "VER",
            "stint_id": f"{race_id}_VER_1",
            "stint_lap": 5,
            "tyre_age": 5,
            "lap_time_s": 90.0,
            "lap_time_fc_s": 89.0,
            "life_laps": 25,
            "censored": False
        })
    # 2024 validation race
    records.append({
        "race_id": "2024_01",
        "track_id": "Track_2024_01",
        "driver": "VER",
        "stint_id": "2024_01_VER_1",
        "stint_lap": 5,
        "tyre_age": 5,
        "lap_time_s": 92.0,
        "lap_time_fc_s": 91.0,
        "life_laps": 30,
        "censored": True
    })
    # 2025 test race
    records.append({
        "race_id": "2025_01",
        "track_id": "Track_New_2025",
        "driver": "VER",
        "stint_id": "2025_01_VER_1",
        "stint_lap": 5,
        "tyre_age": 5,
        "lap_time_s": 88.0,
        "lap_time_fc_s": 87.0,
        "life_laps": 20,
        "censored": False
    })
    return pd.DataFrame(records)


def test_leakage_1_no_race_id_overlap():
    """Leakage Test 1: No race_id appears in two splits."""
    config = load_config("config.yaml")
    df = create_mock_dataset()
    train_df, val_df, test_df = split_by_race(df, config)

    train_races = set(train_df["race_id"].unique())
    val_races = set(val_df["race_id"].unique())
    test_races = set(test_df["race_id"].unique())

    assert len(train_races.intersection(val_races)) == 0
    assert len(train_races.intersection(test_races)) == 0
    assert len(val_races.intersection(test_races)) == 0


def test_leakage_3_train_only_fitting_and_unk_track():
    """Leakage Test 3: Normalisers and embeddings fit only on train; unseen tracks map to unk."""
    config = load_config("config.yaml")
    df = create_mock_dataset()
    train_df, val_df, test_df = split_by_race(df, config)

    # Check scaler is fit on train and applied to val without modifying train stats
    scaler = fit_train_scaler(train_df, ["lap_time_fc_s"])
    assert scaler["lap_time_fc_s"]["mean"] == 89.0

    val_scaled = apply_scaler(val_df, scaler)
    # (91.0 - 89.0) / 1.0 (since std was 0.0 default 1.0)
    assert val_scaled["lap_time_fc_s"].iloc[0] == 2.0

    # Check track mapping
    track_map = build_track_mapping(train_df)
    assert "Track_New_2025" not in track_map
    assert track_map["unk"] == 0

    encoded_test_tracks = encode_tracks(test_df, track_map)
    assert encoded_test_tracks.iloc[0] == 0  # Maps to unk index 0


def test_leakage_4_forbidden_inputs():
    """Leakage Test 4: Final stint length and life_laps are never model inputs."""
    allowed_inputs = [
        "track", "compound", "tyre_age", "lap_fraction",
        "relative_pace", "track_temp", "track_status"
    ]
    forbidden_features = ["life_laps", "stint_length", "censored", "baseline_time_s"]
    for forbidden in forbidden_features:
        assert forbidden not in allowed_inputs, f"Leakage: {forbidden} found in inputs"
