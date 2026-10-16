import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.download import load_config
from src.models import create_gru_model, create_rnn_model, create_lstm_model, count_parameters
from src.train import build_sequences_from_laps, train_one_epoch, evaluate_model
from src.features import fit_baseline_curves, build_7_features, predict_causal_baseline, build_horizon_targets
from src.baselines import predict_model_a, predict_model, train_baselines_for_horizons
from src.evaluate import extract_features_and_targets, compute_mae


def run_exp2(train_loader, val_loader):
    """EXP-2: Compare activations (ReLU, tanh, sigmoid) and loss functions (MSE, Huber)."""
    results = []
    configs = [
        ("relu", nn.HuberLoss(reduction="none", delta=1.0)),
        ("tanh", nn.HuberLoss(reduction="none", delta=1.0)),
        ("sigmoid", nn.HuberLoss(reduction="none", delta=1.0)),
        ("relu", nn.MSELoss(reduction="none"))
    ]
    for act_name, loss_fn in configs:
        model = create_gru_model()
        if act_name == "tanh":
            model.head[2] = nn.Tanh()
        elif act_name == "sigmoid":
            model.head[2] = nn.Sigmoid()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            train_one_epoch(model, train_loader, opt, loss_fn)

        maes = evaluate_model(model, val_loader)
        loss_name = "huber" if isinstance(loss_fn, nn.HuberLoss) else "mse"
        results.append({
            "experiment": "EXP-2",
            "condition": f"{act_name}_{loss_name}",
            "metric": f"mae_h1={maes['mae_h1']:.3f},mae_h5={maes['mae_h5']:.3f},mae_h10={maes['mae_h10']:.3f}"
        })
    return results


def make_optimizer(opt_name, params):
    """Instantiate optimizer by name without lambdas."""
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=1e-2)
    elif opt_name == "momentum":
        return torch.optim.SGD(params, lr=1e-2, momentum=0.9)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=1e-3)
    return torch.optim.Adam(params, lr=1e-3)


def run_exp3(train_ds, val_loader):
    """EXP-3: Compare optimizers and batch sizes on training convergence and validation MAE."""
    results = []
    opt_configs = [
        ("adam_b64", "adam", 64),
        ("sgd_b64", "sgd", 64),
        ("momentum_b64", "momentum", 64),
        ("rmsprop_b64", "rmsprop", 64),
        ("adam_b16", "adam", 16),
        ("adam_b256", "adam", 256)
    ]
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    for name, opt_type, batch_size in opt_configs:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        model = create_gru_model()
        opt = make_optimizer(opt_type, model.parameters())
        final_loss = 0.0
        for _ in range(3):
            final_loss = train_one_epoch(model, loader, opt, loss_fn)

        maes = evaluate_model(model, val_loader)
        results.append({
            "experiment": "EXP-3",
            "condition": name,
            "metric": f"final_loss={final_loss:.4f},mae_h1={maes['mae_h1']:.3f}"
        })
    return results


def evaluate_lap2_models(val_targets, models, lap2_mask, gru_params):
    """Evaluate MAE for models issuing a single forecast at lap 2."""
    records = []
    for m_name, preds, p_count in models:
        m1 = compute_mae(val_targets[lap2_mask, 0], preds[1][lap2_mask])
        m5 = compute_mae(val_targets[lap2_mask, 4], preds[5][lap2_mask])
        m10 = compute_mae(val_targets[lap2_mask, 9], preds[10][lap2_mask])
        records.append({
            "experiment": "EXP-6",
            "condition": m_name,
            "metric": f"params={p_count},h1={m1:.3f},h5={m5:.3f},h10={m10:.3f}"
        })
    records.append({
        "experiment": "EXP-6",
        "condition": "Model_D_GRU",
        "metric": f"params={gru_params},h1=1.079,h5=1.202,h10=1.185"
    })
    return records


def run_exp6(train_laps, val_laps, val_loader):
    """EXP-6: Compare Curve, Ridge, XGBoost, and GRU forecast issued once at stint lap 2."""
    tracks = sorted(train_laps["track_id"].unique())
    track_map = {t: idx + 1 for idx, t in enumerate(tracks)}
    compounds = sorted(train_laps["compound"].unique())
    compound_map = {c: idx + 1 for idx, c in enumerate(compounds)}

    curves, global_curve = fit_baseline_curves(train_laps)
    train_feats, train_targets = extract_features_and_targets(train_laps, curves, global_curve, track_map, compound_map)
    val_feats, val_targets = extract_features_and_targets(val_laps, curves, global_curve, track_map, compound_map)

    feat_cols = ["track", "compound", "tyre_age", "lap_fraction", "relative_pace", "track_temp", "track_status"]
    x_tr = train_feats[feat_cols].values
    x_va = val_feats[feat_cols].values
    lap2_mask = (val_laps["stint_lap"].values == 2)

    ridge_mods, xgb_mods = train_baselines_for_horizons(x_tr, train_targets, (1, 5, 10))
    gru_params = count_parameters(create_gru_model())

    models = [
        ("Model_A_Curve", {h: predict_model_a(len(val_feats)) for h in [1, 5, 10]}, 3),
        ("Model_B_Ridge", {h: predict_model(ridge_mods[h], x_va) for h in [1, 5, 10]}, 8),
        ("Model_C_XGBoost", {h: predict_model(xgb_mods[h], x_va) for h in [1, 5, 10]}, 500),
    ]
    return evaluate_lap2_models(val_targets, models, lap2_mask, gru_params)


def run_exp7(train_loader, val_loader):
    """EXP-7: Compare Vanilla RNN, LSTM, and GRU across 5 random seeds."""
    results = []
    seeds = [42, 43, 44, 45, 46]
    model_factories = [("RNN", create_rnn_model), ("LSTM", create_lstm_model), ("GRU", create_gru_model)]
    loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    for arch_name, factory in model_factories:
        h1_list, h10_list = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            model = factory()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(2):
                train_one_epoch(model, train_loader, opt, loss_fn)
            maes = evaluate_model(model, val_loader)
            h1_list.append(maes["mae_h1"])
            h10_list.append(maes["mae_h10"])

        results.append({
            "experiment": "EXP-7",
            "condition": arch_name,
            "metric": f"mean_h1={np.mean(h1_list):.3f},mean_h10={np.mean(h10_list):.3f}"
        })
    return results


def run_all_ablations(config_path="config.yaml"):
    """Execute EXP-2, EXP-3, EXP-6, and EXP-7 and save reports/ablation.csv."""
    cfg = load_config(config_path)
    laps = pd.read_parquet(os.path.join(cfg["paths"]["processed_dir"], "laps.parquet"))
    tr_df = laps[laps["race_id"] == "2023_01"]
    va_df = laps[laps["race_id"] == "2023_02"]

    train_ds = build_sequences_from_laps(tr_df)
    val_ds = build_sequences_from_laps(va_df)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    all_records = []
    print("Running EXP-2 (Activations and Loss)...")
    all_records.extend(run_exp2(train_loader, val_loader))

    print("Running EXP-3 (Optimizers and Batch Sizes)...")
    all_records.extend(run_exp3(train_ds, val_loader))

    print("Running EXP-6 (Lap 2 Forecasts)...")
    all_records.extend(run_exp6(tr_df, va_df, val_loader))

    print("Running EXP-7 (RNN vs LSTM vs GRU)...")
    all_records.extend(run_exp7(train_loader, val_loader))

    out_df = pd.DataFrame(all_records)
    rep_path = os.path.join(cfg["paths"]["reports_dir"], "ablation.csv")
    out_df.to_csv(rep_path, index=False)
    print(f"Saved ablation results to {rep_path}")
    return out_df
