import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.download import load_config
from src.features import fit_baseline_curves, predict_causal_baseline, build_stint_features, build_horizon_targets
from src.models import create_gru_model, count_parameters


class StintSequenceDataset(Dataset):
    """Dataset yielding sliding sequence windows of stint laps and 20-lap targets."""
    def __init__(self, sequences, targets):
        self.num_x = torch.tensor(sequences["numeric"], dtype=torch.float32)
        self.trk = torch.tensor(sequences["track"], dtype=torch.long)
        self.cmp = torch.tensor(sequences["compound"], dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.mask = ~torch.isnan(self.targets)
        self.targets[~self.mask] = 0.0

    def __len__(self):
        return len(self.num_x)

    def __getitem__(self, idx):
        return self.num_x[idx], self.trk[idx], self.cmp[idx], self.targets[idx], self.mask[idx]


def extract_padded_windows(full_numeric, feats, targets, seq_len=10):
    """Slice and pad sequential sliding windows for a single stint."""
    num_wins, trk_wins, cmp_wins, tgt_wins = [], [], [], []
    num_laps = len(full_numeric)
    for t in range(1, num_laps + 1):
        start_idx = max(0, t - seq_len)
        pad_len = seq_len - (t - start_idx)
        win_num = full_numeric[start_idx:t]
        win_trk = feats["track"].values[start_idx:t]
        win_cmp = feats["compound"].values[start_idx:t]
        if pad_len > 0:
            win_num = np.pad(win_num, ((pad_len, 0), (0, 0)), mode="constant")
            win_trk = np.pad(win_trk, (pad_len, 0), mode="constant")
            win_cmp = np.pad(win_cmp, (pad_len, 0), mode="constant")
        num_wins.append(win_num)
        trk_wins.append(win_trk)
        cmp_wins.append(win_cmp)
        tgt_wins.append(targets[t - 1])
    return num_wins, trk_wins, cmp_wins, tgt_wins


def resolve_maps_and_curves(laps_df, curves, global_curve, track_map, compound_map):
    """Resolve curves and categorical mappings if not provided."""
    if curves is None or global_curve is None:
        curves, global_curve = fit_baseline_curves(laps_df)
    if track_map is None:
        tracks = sorted(laps_df["track_id"].unique())
        track_map = {t: idx + 1 for idx, t in enumerate(tracks)}
    if compound_map is None:
        compounds = sorted(laps_df["compound"].unique())
        compound_map = {c: idx + 1 for idx, c in enumerate(compounds)}
    return curves, global_curve, track_map, compound_map


def build_sequences_from_laps(laps_df, seq_len=10, curves=None, global_curve=None, track_map=None, compound_map=None):
    """Build fixed-window input sequences and horizon targets from cleaned laps."""
    curves, global_curve, track_map, compound_map = resolve_maps_and_curves(
        laps_df, curves, global_curve, track_map, compound_map
    )
    num_list, trk_list, cmp_list, tgt_list = [], [], [], []
    num_cols = [
        "tyre_age", "lap_fraction", "relative_pace", "track_temp",
        "track_status", "gap_mask", "track_temp_delta_c"
    ]

    for stint_id in laps_df["stint_id"].unique():
        sub = laps_df[laps_df["stint_id"] == stint_id].sort_values("stint_lap")
        feats = build_stint_features(sub, track_map, compound_map, include_temp_delta=True)
        baseline = predict_causal_baseline(sub, curves, global_curve)
        targets = build_horizon_targets(sub, baseline, horizon=20)
        curr_res = (sub["lap_time_fc_s"].values - baseline).reshape(-1, 1)
        full_numeric = np.hstack([feats[num_cols].values, curr_res])

        nw, tw, cw, tg = extract_padded_windows(full_numeric, feats, targets, seq_len)
        num_list.extend(nw)
        trk_list.extend(tw)
        cmp_list.extend(cw)
        tgt_list.extend(tg)

    seqs = {"numeric": np.array(num_list), "track": np.array(trk_list), "compound": np.array(cmp_list)}
    return StintSequenceDataset(seqs, np.array(tgt_list))


def asymmetric_huber_loss(pred, target, delta=1.0, positive_weight=0.5):
    """Huber loss with asymmetric weighting for slower-than-predicted laps."""
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(delta, device=err.device))
    lin = abs_err - quad
    base_loss = 0.5 * (quad ** 2) + delta * lin
    weights = torch.where(err < 0, torch.tensor(positive_weight, device=err.device), torch.tensor(1.0, device=err.device))
    return weights * base_loss


def masked_loss(pred, target, mask, loss_fn):
    """Compute loss strictly on non-NaN horizon targets."""
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    loss = loss_fn(pred, target)
    return (loss * mask).sum() / mask.sum()


def train_one_epoch(model, loader, optimizer, loss_fn, clip=1.0):
    """Execute one training epoch with gradient clipping."""
    model.train()
    total_loss, total_steps = 0.0, 0
    for num_x, trk, cmp, targets, mask in loader:
        optimizer.zero_grad()
        out = model(num_x, trk, cmp)
        loss = masked_loss(out, targets, mask, loss_fn)
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
    return total_loss / max(total_steps, 1)


def evaluate_model(model, loader):
    """Compute validation MAE at horizons 1, 5, and 10."""
    model.eval()
    errors = {1: [], 5: [], 10: []}
    with torch.no_grad():
        for num_x, trk, cmp, targets, mask in loader:
            out = model(num_x, trk, cmp)
            for h in [1, 5, 10]:
                val_mask = mask[:, h - 1]
                if val_mask.sum() > 0:
                    err = torch.abs(out[val_mask, h - 1] - targets[val_mask, h - 1])
                    errors[h].extend(err.cpu().numpy().tolist())

    return {f"mae_h{h}": float(np.mean(errors[h])) if errors[h] else np.nan for h in [1, 5, 10]}


def setup_training_datasets(train_laps, val_laps, seq_len=10):
    """Build datasets using train-fitted curves and vocabularies."""
    curves, global_curve = fit_baseline_curves(train_laps)
    tracks = sorted(train_laps["track_id"].unique())
    track_map = {t: idx + 1 for idx, t in enumerate(tracks)}
    compounds = sorted(train_laps["compound"].unique())
    compound_map = {c: idx + 1 for idx, c in enumerate(compounds)}
    tr_ds = build_sequences_from_laps(train_laps, seq_len, curves, global_curve, track_map, compound_map)
    va_ds = build_sequences_from_laps(val_laps, seq_len, curves, global_curve, track_map, compound_map)
    return tr_ds, va_ds


def setup_pilot_loaders(train_laps, val_laps, batch_size=64):
    """Build datasets and data loaders using train-fitted curves and vocabularies."""
    tr_ds, va_ds = setup_training_datasets(train_laps, val_laps, seq_len=10)
    return DataLoader(tr_ds, batch_size=batch_size, shuffle=True), DataLoader(va_ds, batch_size=batch_size, shuffle=False)


def run_pilot(train_laps, val_laps, run_name="pilot", epochs=3, seed=42):
    """Run a small pilot training on CPU and record metrics in runs/."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, val_loader = setup_pilot_loaders(train_laps, val_laps, batch_size=64)
    model = create_gru_model(num_numeric=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {loss:.4f}")

    metrics = evaluate_model(model, val_loader)
    metrics["epochs"], metrics["param_count"] = epochs, count_parameters(model)
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)
    print(f"Pilot run completed. Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TyreSense GRU training script")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--run-name", type=str, default="pilot")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config("config.yaml")
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    tr = laps[laps["race_id"] == "2023_01"]
    va = laps[laps["race_id"] == "2023_02"]
    run_pilot(tr, va, run_name=args.run_name, epochs=args.epochs, seed=args.seed)
