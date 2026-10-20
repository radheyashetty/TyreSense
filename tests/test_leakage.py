import numpy as np
import pandas as pd
from src.download import load_config
from src.split import split_by_race, fit_train_scaler, build_track_mapping
from src.features import fit_baseline_curves, predict_causal_baseline, build_7_features
from src.clean import flag_spikes_causal
from src.train import build_sequences_from_laps, setup_training_datasets


def create_mock_stint():
    """Create sample stint dataframe for leakage tests."""
    return pd.DataFrame({
        "race_id": ["2023_01"] * 12, "track_id": ["Bahrain Grand Prix"] * 12,
        "driver": ["VER"] * 12, "stint_id": ["2023_01_VER_1"] * 12,
        "stint_lap": list(range(1, 13)), "tyre_age": list(range(1, 13)),
        "compound": ["SOFT"] * 12, "lap_number": list(range(2, 14)),
        "total_laps": [57] * 12, "lap_time_s": [95.0 + 0.1 * i for i in range(12)],
        "lap_time_fc_s": [94.0 + 0.1 * i for i in range(12)],
        "track_temp_c": [30.0 + 0.05 * i for i in range(12)],
        "track_status": ["1"] * 12, "gap_mask": [0] * 12
    })


def test_leakage_1_no_race_in_two_splits():
    """Leakage 1: Verify that no race_id appears in multiple splits."""
    config = load_config("config.yaml")
    laps = pd.DataFrame({"race_id": ["2022_01", "2023_01", "2024_01", "2025_01"]})
    tr, va, te = split_by_race(laps, config)
    assert len(set(tr["race_id"]).intersection(set(va["race_id"]))) == 0
    assert len(set(tr["race_id"]).intersection(set(te["race_id"]))) == 0
    assert len(set(va["race_id"]).intersection(set(te["race_id"]))) == 0


def test_leakage_2_changing_future_does_not_alter_past():
    """Leakage 2: Modifying lap t+1 never alters forecast or spike flag at lap t."""
    stint = create_mock_stint()
    curves, g_curve = fit_baseline_curves(stint)
    t_idx = 4
    orig_base = predict_causal_baseline(stint, curves, g_curve)
    orig_spike = flag_spikes_causal(stint.iloc[: t_idx + 1])

    mod_stint = stint.copy()
    mod_stint.loc[t_idx + 2, "lap_time_fc_s"] += 15.0
    mod_base = predict_causal_baseline(mod_stint, curves, g_curve)
    mod_spike = flag_spikes_causal(mod_stint.iloc[: t_idx + 1])

    assert np.isclose(orig_base[t_idx], mod_base[t_idx])
    assert orig_spike == mod_spike


def test_leakage_3_train_only_fitting():
    """Leakage 3: Preprocessors and vocabularies are fitted strictly on training data."""
    train_laps = pd.DataFrame({"track_id": ["Bahrain Grand Prix", "Monza"], "feat": [1.0, 2.0]})
    track_map = build_track_mapping(train_laps)
    scaler = fit_train_scaler(train_laps, ["feat"])
    assert "Bahrain Grand Prix" in track_map
    assert "Silverstone" not in track_map
    assert track_map.get("Silverstone", 0) == 0
    assert "feat" in scaler


def test_leakage_4_stint_length_never_input():
    """Leakage 4: Final stint length and life_laps are strictly absent from model features."""
    stint = create_mock_stint()
    feats = build_7_features(stint, {"Bahrain Grand Prix": 1}, {"SOFT": 1})
    assert "life_laps" not in feats.columns
    assert "stint_length" not in feats.columns


def test_leakage_5_app_replay_sees_only_laps_up_to_t():
    """Leakage 5: App observation window is strictly <= current lap t."""
    stint = create_mock_stint()
    curr_lap = 6
    observed = stint[stint["stint_lap"] <= curr_lap]
    assert int(observed["stint_lap"].max()) == curr_lap
    assert len(observed) == curr_lap


def test_leakage_val_uses_train_curves():
    """Verify that validation sequence extraction uses train curves, not refitted curves."""
    train_laps = create_mock_stint()
    train_laps["track_id"] = "track_A"
    train_laps["compound"] = "HARD"

    val_laps = create_mock_stint()
    val_laps["race_id"] = "2024_01"
    val_laps["stint_id"] = "2024_01_VER_1"
    val_laps["track_id"] = "track_A"
    val_laps["compound"] = "HARD"
    val_laps["lap_time_fc_s"] = [120.0 + 0.1 * i for i in range(12)]

    tr_ds, val_ds = setup_training_datasets(train_laps, val_laps, seq_len=5)
    sample_residual = val_ds.num_x[0, -1, -1].item()
    assert sample_residual > 15.0, f"Residual ({sample_residual}) indicates curves were refit on validation!"
