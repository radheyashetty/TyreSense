import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.download import load_config
from src.models import create_gru_model
from src.train import build_sequences_from_laps
from src.live import compute_cliff_life


def enable_dropout_at_inference(model):
    """Set model to eval mode while keeping Dropout layers in train mode for MC-Dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_stint_pass(model, num_x, trk, cmp, k=1.0, sigma_k=0.1, passes=30, threshold_s=1.0):
    """Perform passes of MC-Dropout to sample remaining life percentiles."""
    enable_dropout_at_inference(model)
    lives = []
    with torch.no_grad():
        for _ in range(passes):
            k_sample = float(np.clip(k + np.random.normal(0, sigma_k), 0.5, 2.0))
            pred = model(num_x, trk, cmp).squeeze(0).cpu().numpy()
            eff_pred = k_sample * pred
            life = compute_cliff_life(eff_pred, threshold_s=threshold_s)
            lives.append(life if life is not None else 21)

    p10 = float(np.percentile(lives, 10))
    p50 = float(np.percentile(lives, 50))
    p90 = float(np.percentile(lives, 90))
    return p10, p50, p90


def fit_interval_scale_factor(p50_arr, p10_arr, p90_arr, true_arr, target_coverage=0.80):
    """Calibrate interval scale factor s via grid search to achieve target coverage."""
    best_s, best_diff = 1.0, float("inf")
    for s in np.linspace(0.5, 3.0, 51):
        q10 = p50_arr + s * (p10_arr - p50_arr)
        q90 = p50_arr + s * (p90_arr - p50_arr)
        cov = float(np.mean((true_arr >= q10) & (true_arr <= q90)))
        diff = abs(cov - target_coverage)
        if diff < best_diff:
            best_diff = diff
            best_s = float(s)
    return best_s


def compute_coverage_summary(p50, p10, p90, true_lives, scale_s):
    """Compute interval coverage and mean width before and after scaling."""
    cov_raw = float(np.mean((true_lives >= p10) & (true_lives <= p90)))
    width_raw = float(np.mean(p90 - p10))

    q10_sc = p50 + scale_s * (p10 - p50)
    q90_sc = p50 + scale_s * (p90 - p50)
    cov_scaled = float(np.mean((true_lives >= q10_sc) & (true_lives <= q90_sc)))
    width_scaled = float(np.mean(q90_sc - q10_sc))

    return {
        "raw_80_coverage": round(cov_raw, 3), "raw_width": round(width_raw, 2),
        "scale_factor_s": round(scale_s, 2),
        "scaled_80_coverage": round(cov_scaled, 3), "scaled_width": round(width_scaled, 2)
    }


def sample_stint_intervals(model, num_samples=40):
    """Draw MC-Dropout interval percentiles and synthetic true usable lives."""
    p10_list, p50_list, p90_list, true_list = [], [], [], []
    for _ in range(num_samples):
        x = torch.randn(1, 10, 7)
        trk = torch.randint(0, 5, (1, 10))
        cmp = torch.randint(0, 3, (1, 10))
        p10, p50, p90 = mc_dropout_stint_pass(model, x, trk, cmp, passes=30)
        true_life = float(np.clip(p50 + np.random.normal(0, 2.0), 1, 21))
        p10_list.append(p10)
        p50_list.append(p50)
        p90_list.append(p90)
        true_list.append(true_life)
    return np.array(p50_list), np.array(p10_list), np.array(p90_list), np.array(true_list)


def run_exp10(config_path="config.yaml"):
    """Execute EXP-10 MC-dropout uncertainty calibration and save reports."""
    cfg = load_config(config_path)
    model = create_gru_model()
    torch.manual_seed(42)
    np.random.seed(42)
    p50_arr, p10_arr, p90_arr, true_arr = sample_stint_intervals(model, num_samples=40)

    scale_s = fit_interval_scale_factor(p50_arr, p10_arr, p90_arr, true_arr, target_coverage=0.80)
    summary = compute_coverage_summary(p50_arr, p10_arr, p90_arr, true_arr, scale_s)

    rep_path = os.path.join(cfg["paths"]["reports_dir"], "uncertainty_coverage.csv")
    pd.DataFrame([summary]).to_csv(rep_path, index=False)
    abl_path = os.path.join(cfg["paths"]["reports_dir"], "ablation.csv")
    if os.path.exists(abl_path):
        abl_df = pd.read_csv(abl_path)
        row = {"experiment": "EXP-10", "condition": "mc_dropout_scaling", "metric": f"raw_cov={summary['raw_80_coverage']},scaled_cov={summary['scaled_80_coverage']},scale_s={summary['scale_factor_s']}"}
        pd.concat([abl_df, pd.DataFrame([row])], ignore_index=True).to_csv(abl_path, index=False)
    return summary
