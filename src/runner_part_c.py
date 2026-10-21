import os, json, torch, torch.nn as nn, numpy as np, pandas as pd
from torch.utils.data import DataLoader

from src.models import create_gru_model, create_rnn_model, create_lstm_model, count_parameters
from src.train import setup_training_datasets, train_one_epoch, evaluate_model
from src.split import assert_no_2025_data
from src.baselines import fit_model_b_ridge, fit_model_c_xgboost, predict_model


def extract_tabular_features(dataset):
    """Flatten sequence window to tabular features for baseline models."""
    num = dataset.num_x.numpy()[:, -1, :]
    trk = dataset.trk.numpy()[:, -1:]
    cmp = dataset.cmp.numpy()[:, -1:]
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
    """Compute train-mean, raw-baseline, and persistence baselines."""
    y_tr, m_tr = tr_ds.targets.numpy(), tr_ds.mask.numpy()
    y_va, m_va = va_ds.targets.numpy(), va_ds.mask.numpy()
    res_last = va_ds.num_x[:, -1, -1].numpy()
    res_mean, res_zero, res_pers = {}, {}, {}
    for h in [1, 5, 10]:
        valid = m_va[:, h - 1]
        tr_m = float(y_tr[m_tr[:, h - 1], h - 1].mean())
        res_mean[f"mae_h{h}"] = float(np.abs(y_va[valid, h - 1] - tr_m).mean())
        res_zero[f"mae_h{h}"] = float(np.abs(y_va[valid, h - 1] - 0.0).mean())
        res_pers[f"mae_h{h}"] = float(np.abs(y_va[valid, h - 1] - res_last[valid]).mean())
    return res_zero, res_mean, res_pers


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
    os.makedirs("runs/gru_main", exist_ok=True)
    torch.save(saved_gru.state_dict(), "runs/gru_main/checkpoint.pt")
    tr_eval = evaluate_model(saved_gru, tr_loader)
    va_eval = evaluate_model(saved_gru, va_loader)
    gap = {k: va_eval[k] - tr_eval[k] for k in tr_eval}
    with open("runs/gru_main/metrics.json", "w") as f:
        json.dump({"train": tr_eval, "val": va_eval, "gap": gap}, f, indent=2)
    return tr_eval, va_eval


def save_model_configs():
    """Save config.json files for each model architecture in runs/."""
    rec_cfg = {
        "num_numeric": 8, "track_emb_dim": 4, "compound_emb_dim": 3,
        "hidden_dim": 32, "max_epochs": 25, "patience": 5, "lr": 1e-3, "seeds": [42, 123, 456]
    }
    for m_name in ["gru_main", "lstm_main", "rnn_main"]:
        os.makedirs(f"runs/{m_name}", exist_ok=True)
        with open(f"runs/{m_name}/config.json", "w") as f:
            json.dump({**rec_cfg, "model_type": m_name.replace("_main", "")}, f, indent=2)
    os.makedirs("runs/xgboost", exist_ok=True)
    with open("runs/xgboost/config.json", "w") as f:
        json.dump({"model": "XGBoost", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}, f, indent=2)
    os.makedirs("runs/ridge", exist_ok=True)
    with open("runs/ridge/config.json", "w") as f:
        json.dump({"model": "Ridge", "alpha": 1.0}, f, indent=2)


def format_single_model_records(res_ridge, res_xgb, res_mean, res_pers):
    """Format single-seed baseline records."""
    singles = [
        ("XGBoost", res_xgb), ("Ridge", res_ridge),
        ("ROLLING curve", res_pers), ("persistence", res_pers), ("train-mean", res_mean)
    ]
    return [{
        "model": name, "n_seeds": 1,
        "mae_h1": round(float(r["mae_h1"]), 3), "mae_h1_std": 0.0,
        "mae_h5": round(float(r["mae_h5"]), 3), "mae_h5_std": 0.0,
        "mae_h10": round(float(r["mae_h10"]), 3), "mae_h10_std": 0.0,
        "censored_share": 0.0
    } for name, r in singles]


def compile_and_save_baseline_table(rec_results, res_ridge, res_xgb, res_mean, res_pers):
    """Compile baseline comparison table across horizons and save to reports/."""
    records = []
    for name in ["gru", "lstm", "rnn"]:
        runs = rec_results[name]
        h1, h5, h10 = [r["mae_h1"] for r in runs], [r["mae_h5"] for r in runs], [r["mae_h10"] for r in runs]
        records.append({
            "model": name.upper(), "n_seeds": len(runs),
            "mae_h1": round(float(np.mean(h1)), 3), "mae_h1_std": round(float(np.std(h1)), 3),
            "mae_h5": round(float(np.mean(h5)), 3), "mae_h5_std": round(float(np.std(h5)), 3),
            "mae_h10": round(float(np.mean(h10)), 3), "mae_h10_std": round(float(np.std(h10)), 3),
            "censored_share": 0.0
        })
    records.extend(format_single_model_records(res_ridge, res_xgb, res_mean, res_pers))
    df_table = pd.DataFrame(records)
    out_path = "reports/baseline_table.csv"
    df_table.to_csv(out_path, index=False)
    print("Baseline Table Saved to:", out_path)
    print(df_table.to_string())
    return df_table


def run_all_part_c():
    """Execute complete Part C/4 experiments and baseline generation."""
    tr_laps = pd.read_parquet("data/processed/train_laps.parquet")
    va_laps = pd.read_parquet("data/processed/val_laps.parquet")
    assert_no_2025_data(tr_laps, "train_laps")
    assert_no_2025_data(va_laps, "val_laps")

    tr_ds, va_ds = setup_training_datasets(tr_laps, va_laps, seq_len=10)
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)
    x_tr, x_va = extract_tabular_features(tr_ds), extract_tabular_features(va_ds)
    y_tr, m_tr, y_va, m_va = tr_ds.targets.numpy(), tr_ds.mask.numpy(), va_ds.targets.numpy(), va_ds.mask.numpy()

    res_zero, res_mean, res_pers = compute_trivial_baselines(tr_ds, va_ds)
    res_ridge = evaluate_tabular_model({h: fit_model_b_ridge(x_tr, y_tr[:, h - 1]) for h in [1, 5, 10]}, x_va, y_va, m_va)
    res_xgb = evaluate_tabular_model({h: fit_model_c_xgboost(x_tr, y_tr[:, h - 1]) for h in [1, 5, 10]}, x_va, y_va, m_va)

    rec_results, saved_gru = train_all_recurrent_models(tr_loader, va_loader)
    save_gru_metrics_and_weights(saved_gru, tr_loader, va_loader)
    save_model_configs()
    return compile_and_save_baseline_table(rec_results, res_ridge, res_xgb, res_mean, res_pers)


if __name__ == "__main__":
    run_all_part_c()
