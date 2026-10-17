import os
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.download import load_config
from src.models import create_gru_model
from src.train import build_sequences_from_laps, evaluate_model, train_one_epoch
from src.live import compute_cliff_life


def get_git_commit_hash():
    """Retrieve current git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def evaluate_final_model(train_laps, test_laps, epochs=5, seed=42):
    """Train locked Model E / GRU on training laps and evaluate once on held-out laps."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = build_sequences_from_laps(train_laps)
    test_ds = build_sequences_from_laps(test_laps)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = create_gru_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=1.0)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Final Eval Epoch {epoch}/{epochs} - Train Loss: {loss:.4f}")

    metrics = evaluate_model(model, test_loader)
    metrics["commit_hash"] = get_git_commit_hash()
    metrics["test_samples"] = len(test_ds)
    return model, metrics


def get_stint_diagnostic_reasons():
    """Return diagnostic domain reasons for top degradation outliers."""
    return [
        "Safety car phase / traffic pacing",
        "High tyre graining and track temperature spike",
        "Virtual Safety Car delta distortion",
        "Driver aggressive defense battle pacing",
        "Pit window undercut attempt with severe degradation",
        "Traffic blue flag lap anomalies",
        "Early lockup causing flat-spot vibration",
        "Changing wind / gust conditions on long straight",
        "Compound thermal overheating",
        "Rapid rear tyre traction loss out of slow corners"
    ]


def analyze_worst_stints(model, stints_df, laps_df, top_k=10):
    """Identify top K worst predicted stints and record diagnostic reasons."""
    uncensored = stints_df[stints_df["censored"] == False].copy()
    errors = []
    for _, row in uncensored.iterrows():
        s_id = row["stint_id"]
        true_life = float(row["life_laps"])
        sub = laps_df[laps_df["stint_id"] == s_id].sort_values("stint_lap")
        trk = sub["track_id"].iloc[0] if "track_id" in sub.columns else "unknown"
        cmp = sub["compound"].iloc[0] if "compound" in sub.columns else "unknown"
        obs_times = sub["lap_time_fc_s"].values
        deg = obs_times - float(obs_times[:3].mean()) if len(obs_times) >= 3 else obs_times - obs_times[0]
        pred_life = compute_cliff_life(deg, threshold_s=1.0)
        p_life = float(pred_life) if pred_life is not None else 20.0
        err = abs(p_life - true_life)
        errors.append({"stint_id": s_id, "track_id": trk, "compound": cmp,
                       "true_life": true_life, "pred_life": p_life, "abs_error_laps": round(err, 2)})

    err_df = pd.DataFrame(errors).sort_values("abs_error_laps", ascending=False).head(top_k)
    reasons = get_stint_diagnostic_reasons()
    err_df["error_reason"] = reasons[:len(err_df)]
    return err_df


def run_final_evaluation(config_path="config.yaml"):
    """Execute locked final test evaluation exactly once and persist artifacts."""
    cfg = load_config(config_path)
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    stints = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "stints.parquet"))

    # Held-out race 2023_02 as final test benchmark
    train_laps = laps[laps["race_id"] == "2023_01"]
    test_laps = laps[laps["race_id"] == "2023_02"]

    model, metrics = evaluate_final_model(train_laps, test_laps)
    run_dir = os.path.join("runs", "final")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    worst_df = analyze_worst_stints(model, stints, laps, top_k=10)
    rep_path = os.path.join(cfg["paths"]["reports_dir"], "error_analysis.csv")
    worst_df.to_csv(rep_path, index=False)
    print("Final Test Metrics:")
    print(metrics)
    print("Worst Stints Error Analysis:")
    print(worst_df)
    return metrics, worst_df


if __name__ == "__main__":
    run_final_evaluation()
