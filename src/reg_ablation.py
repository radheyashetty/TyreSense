import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso
from torch.utils.data import DataLoader
from src.download import load_config
from src.models import create_gru_model
from src.train import setup_training_datasets, train_one_epoch, evaluate_model


def run_exp4_l2_sweep(train_loader, val_loader):
    """EXP-4: Evaluate L2 weight decay sweep and train-val gap on GRU."""
    results = []
    l2_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    for wd in l2_values:
        torch.manual_seed(42)
        model = create_gru_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(3):
            train_one_epoch(model, train_loader, opt, loss_fn)
        tr_mae = evaluate_model(model, train_loader)["mae_h1"]
        va_mae = evaluate_model(model, val_loader)["mae_h1"]
        gap = va_mae - tr_mae
        results.append({
            "experiment": "EXP-4", "condition": f"l2_{wd}",
            "metric": f"val_mae_h1={va_mae:.3f},train_val_gap={gap:.3f}"
        })
    return results


def run_exp4_dropout_sweep(train_loader, val_loader):
    """EXP-4: Evaluate dropout rates on GRU generalization."""
    results = []
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.5]
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    for dr in dropouts:
        torch.manual_seed(42)
        model = create_gru_model()
        model.input_drop = nn.Dropout(dr)
        model.head[0] = nn.Dropout(dr)
        model.head[3] = nn.Dropout(dr)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(3):
            train_one_epoch(model, train_loader, opt, loss_fn)
        va_mae = evaluate_model(model, val_loader)["mae_h1"]
        results.append({
            "experiment": "EXP-4", "condition": f"dropout_{dr}",
            "metric": f"val_mae_h1={va_mae:.3f}"
        })
    return results


def run_exp4_lasso_sparsity(train_ds, val_ds):
    """EXP-4: Fit L1 Lasso on linear baseline and count non-zero weights."""
    x_tr = train_ds.num_x[:, -1, :].numpy()
    y_tr = train_ds.targets[:, 0].numpy()
    x_va = val_ds.num_x[:, -1, :].numpy()
    y_va = val_ds.targets[:, 0].numpy()

    results = []
    for alpha in [0.001, 0.01, 0.1]:
        lasso = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        lasso.fit(x_tr, y_tr)
        nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-5))
        preds = lasso.predict(x_va)
        mae = float(np.mean(np.abs(preds - y_va)))
        results.append({
            "experiment": "EXP-4", "condition": f"lasso_alpha_{alpha}",
            "metric": f"nonzero_weights={nonzero}/{len(lasso.coef_)},val_mae={mae:.3f}"
        })
    return results


def run_single_patience_trial(pat, train_loader, val_loader, loss_fn, max_epochs):
    """Run training for a specific patience limit and return best MAE and epochs used."""
    torch.manual_seed(42)
    model = create_gru_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, patience_counter, epochs_used = float("inf"), 0, 0
    for ep in range(1, max_epochs + 1):
        epochs_used = ep
        train_one_epoch(model, train_loader, opt, loss_fn)
        val_mae = evaluate_model(model, val_loader)["mae_h1"]
        if val_mae < best_loss - 1e-4:
            best_loss, patience_counter = val_mae, 0
        else:
            patience_counter += 1
        if pat < max_epochs and patience_counter >= pat:
            break
    return best_loss, epochs_used


def run_exp5_early_stopping(train_loader, val_loader, max_epochs=20):
    """EXP-5: Compare early stopping patience (5, 15, 30) vs fixed epochs."""
    results = []
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    for pat in [5, 15, 30, max_epochs]:
        best_loss, epochs = run_single_patience_trial(pat, train_loader, val_loader, loss_fn, max_epochs)
        cond = f"patience_{pat}" if pat < max_epochs else f"fixed_{max_epochs}_epochs"
        results.append({
            "experiment": "EXP-5", "condition": cond,
            "metric": f"val_mae_h1={best_loss:.3f},epochs_used={epochs}"
        })
    return results


def run_part2_ablations(config_path="config.yaml"):
    """Execute EXP-4 and EXP-5 and append records to reports/ablation.csv."""
    cfg = load_config(config_path)
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    tr_df = laps[laps["race_id"] == "2023_01"]
    va_df = laps[laps["race_id"] == "2023_02"]

    train_ds, val_ds = setup_training_datasets(tr_df, va_df)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    records = []
    records.extend(run_exp4_l2_sweep(train_loader, val_loader))
    records.extend(run_exp4_dropout_sweep(train_loader, val_loader))
    records.extend(run_exp4_lasso_sparsity(train_ds, val_ds))
    records.extend(run_exp5_early_stopping(train_loader, val_loader, max_epochs=25))

    rep_path = os.path.join(cfg["paths"]["reports_dir"], "ablation.csv")
    existing_df = pd.read_csv(rep_path) if os.path.exists(rep_path) else pd.DataFrame()
    combined_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
    combined_df.to_csv(rep_path, index=False)
    print(f"Appended EXP-4 and EXP-5 results to {rep_path}")
    return combined_df
