import numpy as np
import pandas as pd
from src.features import (
    fit_baseline_curves,
    predict_causal_baseline,
    compute_relative_pace,
    build_7_features,
    build_stint_features,
    build_horizon_targets,
)


def create_sample_stint():
    """Create a sample stint of 15 clean laps."""
    return pd.DataFrame({
        "race_id": ["2023_01"] * 15,
        "track_id": ["Bahrain Grand Prix"] * 15,
        "driver": ["VER"] * 15,
        "stint_id": ["2023_01_VER_1"] * 15,
        "stint_lap": list(range(1, 16)),
        "tyre_age": list(range(1, 16)),
        "compound": ["SOFT"] * 15,
        "lap_number": list(range(2, 17)),
        "total_laps": [57] * 15,
        "lap_time_s": [95.0 + 0.1 * i for i in range(15)],
        "lap_time_fc_s": [94.0 + 0.1 * i for i in range(15)],
        "track_temp_c": [30.0 + 0.1 * i for i in range(15)],
        "track_status": ["1"] * 15,
        "gap_mask": [0] * 15,
    })


def test_leakage_2_changing_future_laps_does_not_change_past():
    """Leakage Test 2: Changing lap t+1 or later never changes the forecast or features at lap t."""
    stint_df = create_sample_stint()
    curves, global_curve = fit_baseline_curves(stint_df)
    track_map = {"Bahrain Grand Prix": 1}
    compound_map = {"SOFT": 1}

    orig_baseline = predict_causal_baseline(stint_df, curves, global_curve)
    orig_features = build_7_features(stint_df, track_map, compound_map)

    modified_stint = stint_df.copy()
    modified_stint.loc[9, "lap_time_fc_s"] += 5.0
    modified_stint.loc[9, "track_temp_c"] += 10.0

    new_baseline = predict_causal_baseline(modified_stint, curves, global_curve)
    new_features = build_7_features(modified_stint, track_map, compound_map)

    np.testing.assert_allclose(orig_baseline[:5], new_baseline[:5])
    pd.testing.assert_frame_equal(orig_features.iloc[:5], new_features.iloc[:5])


def test_leakage_5_baseline_curve_is_causal():
    """Leakage Test 5: The baseline curve is strictly causal."""
    stint_df = create_sample_stint()
    curves, global_curve = fit_baseline_curves(stint_df)

    mod_stint = stint_df.copy()
    mod_stint.loc[7, "lap_time_fc_s"] += 10.0

    orig_baseline = predict_causal_baseline(stint_df, curves, global_curve)
    mod_baseline = predict_causal_baseline(mod_stint, curves, global_curve)

    np.testing.assert_allclose(orig_baseline[:3], mod_baseline[:3])


def test_seven_inputs_and_horizon_targets():
    """Verify 7 inputs structure and future horizon target residual generation."""
    stint_df = create_sample_stint()
    curves, global_curve = fit_baseline_curves(stint_df)
    track_map = {"Bahrain Grand Prix": 1}
    compound_map = {"SOFT": 1}

    feats = build_7_features(stint_df, track_map, compound_map)
    expected_cols = [
        "track", "compound", "tyre_age", "lap_fraction",
        "relative_pace", "track_temp", "track_status", "gap_mask"
    ]
    assert list(feats.columns) == expected_cols
    assert len(feats) == 15

    baseline = predict_causal_baseline(stint_df, curves, global_curve)
    targets = build_horizon_targets(stint_df, baseline, horizon=20)
    assert targets.shape == (15, 20)
    expected_res_1 = stint_df["lap_time_fc_s"].iloc[1] - baseline[1]
    assert np.isclose(targets[0, 0], expected_res_1)
    assert np.isnan(targets[14, 0])


def test_track_temp_delta_at_lap_one_is_zero():
    """Verify that track_temp_delta_c is exactly 0.0 at stint lap 1."""
    stint_df = create_sample_stint()
    track_map = {"Bahrain Grand Prix": 1}
    compound_map = {"SOFT": 1}
    feats = build_stint_features(stint_df, track_map, compound_map, include_temp_delta=True)
    assert "track_temp_delta_c" in feats.columns
    assert feats["track_temp_delta_c"].iloc[0] == 0.0
    expected_delta = stint_df["track_temp_c"].iloc[5] - stint_df["track_temp_c"].iloc[0]
    assert np.isclose(feats["track_temp_delta_c"].iloc[5], expected_delta)


def test_perturbing_future_temp_does_not_change_earlier_features():
    """Verify that changing temperature at lap t+1 does not change earlier inputs."""
    stint_df = create_sample_stint()
    track_map = {"Bahrain Grand Prix": 1}
    compound_map = {"SOFT": 1}

    orig_feats = build_stint_features(stint_df, track_map, compound_map, include_temp_delta=True)
    mod_stint = stint_df.copy()
    mod_stint.loc[10, "track_temp_c"] += 15.0

    mod_feats = build_stint_features(mod_stint, track_map, compound_map, include_temp_delta=True)
    pd.testing.assert_frame_equal(orig_feats.iloc[:10], mod_feats.iloc[:10])
