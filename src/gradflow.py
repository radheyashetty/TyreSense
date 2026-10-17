import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def measure_cell_gradients(cell_cls, seq_len, in_dim=14, h_dim=32, batch_size=4):
    """Compute step-wise gradient norms dL/dh_t using unrolled recurrent cells."""
    torch.manual_seed(42)
    cell = cell_cls(in_dim, h_dim)
    head = nn.Linear(h_dim, 20)
    x = torch.randn(batch_size, seq_len, in_dim)
    h = torch.zeros(batch_size, h_dim)
    c = torch.zeros(batch_size, h_dim) if cell_cls == nn.LSTMCell else None
    h_states = []

    for t in range(seq_len):
        if cell_cls == nn.LSTMCell:
            h, c = cell(x[:, t, :], (h, c))
        else:
            h = cell(x[:, t, :], h)
        h.retain_grad()
        h_states.append(h)

    loss = torch.sum(head(h_states[-1]) ** 2)
    loss.backward()

    norms = [float(torch.norm(s.grad).item()) for s in h_states]
    ratio = norms[0] / max(norms[-1], 1e-12)
    return norms, ratio


def evaluate_all_architectures(lengths=(10, 25, 50, 100)):
    """Evaluate gradient flow decay across RNN, LSTM, and GRU over sequence lengths."""
    cells = [("RNN", nn.RNNCell, 58), ("LSTM", nn.LSTMCell, 27), ("GRU", nn.GRUCell, 32)]
    records, norm_curves = [], {}

    for name, cell_cls, h_dim in cells:
        norm_curves[name] = {}
        for l in lengths:
            norms, ratio = measure_cell_gradients(cell_cls, l, in_dim=14, h_dim=h_dim)
            norm_curves[name][l] = norms
            records.append({
                "architecture": name, "seq_len": l,
                "first_norm": round(norms[0], 6), "last_norm": round(norms[-1], 6),
                "ratio_first_to_last": round(ratio, 6)
            })
    return pd.DataFrame(records), norm_curves


def plot_gradient_flow(norm_curves, output_path="reports/figures/gradflow.png"):
    """Plot gradient flow decay curves across architectures for seq_len=50."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    steps = np.arange(1, 51)
    for name in ["RNN", "LSTM", "GRU"]:
        curve = norm_curves[name][50]
        plt.plot(steps, curve, label=name, linewidth=2)

    plt.xlabel("Sequence Time Step t")
    plt.ylabel("Gradient Norm ||dL/dh_t||")
    plt.yscale("log")
    plt.title("EXP-8: Recurrent Gradient Flow (Length 50 Stress Test)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved gradient flow plot to {output_path}")


def check_clipping_stability(clips=(0.5, 1.0, 5.0)):
    """Verify training convergence under different gradient clipping thresholds."""
    statuses = {}
    for c in clips:
        torch.manual_seed(42)
        cell = nn.GRUCell(14, 32)
        opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
        diverged = False
        for _ in range(5):
            x = torch.randn(4, 14)
            h = torch.zeros(4, 32)
            h = cell(x, h)
            loss = torch.sum(h ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell.parameters(), c)
            opt.step()
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
        statuses[c] = "diverged" if diverged else "stable"
    return statuses


def run_exp8(output_dir="reports"):
    """Run EXP-8 gradient flow experiments, save table and figure."""
    df_results, norm_curves = evaluate_all_architectures()
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "gradflow.csv")
    df_results.to_csv(csv_path, index=False)
    fig_path = os.path.join(output_dir, "figures", "gradflow.png")
    plot_gradient_flow(norm_curves, fig_path)

    clip_statuses = check_clipping_stability()
    print("Clipping stability:", clip_statuses)
    print(df_results)
    return df_results
