import numpy as np
import torch
import torch.nn as nn
from src.models import create_gru_model
from src.uncertainty import (
    enable_dropout_at_inference,
    mc_dropout_stint_pass,
    fit_interval_scale_factor,
    compute_coverage_summary,
)


def test_dropout_layers_train_at_inference():
    """Verify that enable_dropout_at_inference keeps only Dropout in train mode."""
    model = create_gru_model()
    enable_dropout_at_inference(model)

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            assert m.training is True
        elif isinstance(m, (nn.Linear, nn.GRU, nn.Embedding)):
            assert m.training is False


def test_mc_dropout_ordering():
    """Verify that p10 <= p50 <= p90 across MC-dropout passes."""
    model = create_gru_model()
    num_x = torch.randn(1, 10, 7)
    trk = torch.randint(0, 5, (1, 10))
    cmp = torch.randint(0, 3, (1, 10))

    p10, p50, p90 = mc_dropout_stint_pass(model, num_x, trk, cmp, passes=15)
    assert p10 <= p50 <= p90


def test_interval_scale_factor_calibration():
    """Verify that interval scaling expands narrow intervals to match target coverage."""
    np.random.seed(42)
    n = 100
    p50 = np.full(n, 15.0)
    p10 = np.full(n, 14.5)  # too narrow: width 1.0
    p90 = np.full(n, 15.5)
    true_lives = np.random.normal(15.0, 1.5, n)  # std is 1.5

    raw_cov = np.mean((true_lives >= p10) & (true_lives <= p90))
    s_best = fit_interval_scale_factor(p50, p10, p90, true_lives, target_coverage=0.80)
    summary = compute_coverage_summary(p50, p10, p90, true_lives, s_best)

    assert s_best > 1.0, f"Scale factor s ({s_best}) did not expand narrow interval"
    assert summary["scaled_80_coverage"] >= raw_cov
    assert 0.70 <= summary["scaled_80_coverage"] <= 0.90
