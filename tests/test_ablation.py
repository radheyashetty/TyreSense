import os
import pandas as pd


def test_ablation_report_structure():
    """Verify that reports/ablation.csv contains valid results for EXP-2, 3, 6, 7."""
    ablation_path = "reports/ablation.csv"
    assert os.path.exists(ablation_path), "Missing reports/ablation.csv"

    df = pd.read_csv(ablation_path)
    assert not df.empty
    assert "experiment" in df.columns
    assert "condition" in df.columns
    assert "metric" in df.columns

    experiments = set(df["experiment"].unique())
    assert {"EXP-2", "EXP-3", "EXP-6", "EXP-7"}.issubset(experiments)
    assert len(df) >= 15
    assert not df["metric"].isna().any()
