import os, json, time, torch, torch.nn as nn, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb

from src.models import create_gru_model, create_rnn_model, create_lstm_model, count_parameters
from src.train import setup_training_datasets, train_one_epoch, evaluate_model
from src.features import fit_baseline_curves
from src.split import assert_no_2025_data
from src.baselines import fit_model_b_ridge, fit_model_c_xgboost, predict_model


def extract_tabular_features(dataset):
    """Flatten sequence window to tabular features for baseline models."""
    num = dataset.num_x.numpy()[:, -1, :]  # last step numeric features [N, 8]
    trk = dataset.trk.numpy()[:, -1:]      # track index [N, 1]
    cmp = dataset.cmp.numpy()[:, -1:]      # compound index [N, 1]
    return np.hstack([num, trk, cmp])


def train_recurrent_with_early_stopping(model, tr_loader, va_loader, max_epochs=25, patience=5, lr=1e-3, seed=42):
    """Train recurrent model with early stopping on validation loss."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    best_val_loss, best_weights, patience_cnt = float("inf"), None, 0

    for ep in range(max_epochs):
        train_one_epoch(model, tr_loader, opt, loss_fn)
        val_losses = []
        with torch.no_grad():
            for num_x, trk, cmp, targets, mask in va_loader:
                out = model(num_x, trk, cmp)
                l = loss_fn(out, targets)
                val_losses.append(l[mask].mean().item())
        va_loss = float(np.mean(val_losses))
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    if best_weights:
        model.load_state_dict(best_weights)
    return model, ep + 1


def evaluate_tabular_model(model_map, x_val, y_val, mask_val):
    """Evaluate Ridge or XGBoost horizon models against validation targets."""
    res = {}
    for h in [1, 5, 10]:
        m = model_map[h]
        preds = predict_model(m, x_val)
        valid = mask_val[:, h - 1]
        err = np.abs(preds[valid] - y_val[valid, h - 1])
        res[f"mae_h{h}"] = float(np.mean(err))
    return res


def compute_trivial_baselines(tr_ds, va_ds):
    """Compute always-train-mean and always-zero (Model A) baselines."""
    y_tr, m_tr = tr_ds.targets.numpy(), tr_ds.mask.numpy()
    y_va, m_va = va_ds.targets.numpy(), va_ds.mask.numpy()
    mean_tr = [y_tr[m_tr[:, h - 1], h - 1].mean() for h in [1, 5, 10]]
    res_mean, res_zero = {}, {}
    for idx, h in enumerate([1, 5, 10]):
        valid = m_va[:, h - 1]
        res_mean[f"mae_h{h}"] = float(np.abs(y_va[valid, h - 1] - mean_tr[idx]).mean())
        res_zero[f"mae_h{h}"] = float(np.abs(y_va[valid, h - 1] - 0.0).mean())
    return res_zero, res_mean


def train_all_recurrent_models(tr_loader, va_loader):
    """Train GRU, LSTM, and RNN forecasters across 3 random seeds."""
    rec_results = {"gru": [], "lstm": [], "rnn": []}
    factories = [("gru", create_gru_model), ("lstm", create_lstm_model), ("rnn", create_rnn_model)]
    saved_gru = None
    for seed in [42, 123, 456]:
        for name, factory in factories:
            model = factory(num_numeric=8)
            model, ep = train_recurrent_with_early_stopping(model, tr_loader, va_loader, max_epochs=25, patience=5, seed=seed)
            m = evaluate_model(model, va_loader)
            m["params"], m["seed"], m["stopped_epoch"] = count_parameters(model), seed, ep
            rec_results[name].append(m)
            if name == "gru" and seed == 42:
                saved_gru = model
    return rec_results, saved_gru


def save_gru_metrics_and_weights(saved_gru, tr_loader, va_loader):
    """Save final GRU checkpoint and record training vs validation generalization gap."""
    os.makedirs("runs/final_gru", exist_ok=True)
    torch.save(saved_gru.state_dict(), "runs/final_gru/checkpoint.pt")
    tr_eval = evaluate_model(saved_gru, tr_loader)
    va_eval = evaluate_model(saved_gru, va_loader)
    gap = {k: va_eval[k] - tr_eval[k] for k in tr_eval}
    with open("runs/final_gru/metrics.json", "w") as f:
        json.dump({"train": tr_eval, "val": va_eval, "gap": gap}, f, indent=2)
    return tr_eval, va_eval


def run_all_part_c():
    """Execute complete Part C experiments and baseline generation."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    assert_no_2025_data(tr_laps, "train_laps")
    assert_no_2025_data(va_laps, "val_laps")

    tr_ds, va_ds = setup_training_datasets(tr_laps, va_laps, seq_len=10)
    tr_loader, va_loader = DataLoader(tr_ds, batch_size=64, shuffle=True), DataLoader(va_ds, batch_size=64, shuffle=False)
    x_tr, x_va = extract_tabular_features(tr_ds), extract_tabular_features(va_ds)
    y_tr, m_tr, y_va, m_va = tr_ds.targets.numpy(), tr_ds.mask.numpy(), va_ds.targets.numpy(), va_ds.mask.numpy()

    # Baselines
    res_mod_a, res_mean = compute_trivial_baselines(tr_ds, va_ds)
    res_ridge = evaluate_tabular_model({h: fit_model_b_ridge(x_tr, y_tr[:, h - 1]) for h in [1, 5, 10]}, x_va, y_va, m_va)
    res_xgb = evaluate_tabular_model({h: fit_model_c_xgboost(x_tr, y_tr[:, h - 1]) for h in [1, 5, 10]}, x_va, y_va, m_va)

    rec_results, saved_gru = train_all_recurrent_models(tr_loader, va_loader)
    tr_eval, va_eval = save_gru_metrics_and_weights(saved_gru, tr_loader, va_loader)
    return res_mod_a, res_ridge, res_xgb, res_mean, rec_results, tr_eval, va_eval
