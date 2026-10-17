import numpy as np
import torch
import torch.nn as nn
from src.mlp_numpy import (
    init_mlp_weights,
    forward_mlp,
    compute_mse_loss,
    backward_mlp,
    check_gradients,
    step_adam,
)


def test_numerical_gradient_check():
    """EXP-1: Maximum relative gradient error for 7-input MLP must be < 1e-6."""
    np.random.seed(42)
    x = np.random.randn(8, 7)
    y = np.random.randn(8, 1)
    params = init_mlp_weights(input_dim=7, seed=42)
    max_rel_error = check_gradients(x, y, params, epsilon=1e-5)
    assert max_rel_error < 1e-6, f"Gradient check failed with error {max_rel_error}"


def test_eight_input_mlp_gradient_check():
    """Verify 8-input MLP has maximum relative gradient error < 1e-6."""
    np.random.seed(42)
    x = np.random.randn(8, 8)
    y = np.random.randn(8, 1)
    params = init_mlp_weights(input_dim=8, seed=42)
    max_rel_error = check_gradients(x, y, params, epsilon=1e-5)
    assert max_rel_error < 1e-6, f"8-input gradient check failed with error {max_rel_error}"


def create_torch_equivalent(params, in_dim=7):
    """Build PyTorch sequential model matching NumPy initial weights."""
    model = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"].squeeze()))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"].squeeze()))
        model[4].weight.copy_(torch.from_numpy(params["W3"].T))
        model[4].bias.copy_(torch.from_numpy(params["b3"].squeeze()))
    return model


def train_step_comparison(x_np, y_np, np_params, pt_model, lr=0.01, steps=10):
    """Run synchronized gradient descent steps on NumPy and PyTorch models."""
    opt = torch.optim.SGD(pt_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_pt, y_pt = torch.from_numpy(x_np), torch.from_numpy(y_np)
    for _ in range(steps):
        _, cache = forward_mlp(x_np, np_params)
        grads = backward_mlp(y_np, cache, np_params)
        for k in np_params:
            np_params[k] -= lr * grads[k]
        opt.zero_grad()
        loss = loss_fn(pt_model(x_pt), y_pt)
        loss.backward()
        opt.step()


def test_numpy_vs_pytorch_loss_gap():
    """EXP-1: NumPy MLP loss matches PyTorch loss within 5% after training."""
    np.random.seed(42)
    x_np = np.random.randn(32, 8).astype(np.float32)
    y_np = np.random.randn(32, 1).astype(np.float32)
    params = init_mlp_weights(input_dim=8, seed=42)
    np_params = {k: v.copy() for k, v in params.items()}
    pt_model = create_torch_equivalent(params, in_dim=8)

    train_step_comparison(x_np, y_np, np_params, pt_model, lr=0.01, steps=10)
    final_np_loss = compute_mse_loss(forward_mlp(x_np, np_params)[0], y_np)
    final_pt_loss = float(nn.MSELoss()(pt_model(torch.from_numpy(x_np)), torch.from_numpy(y_np)).item())
    gap = abs(final_np_loss - final_pt_loss) / final_pt_loss
    assert gap < 0.05, f"Loss gap was {gap * 100:.2f}% (exceeds 5%)"


def test_adam_optimizer_decreases_loss():
    """Verify that Adam optimizer updates decrease loss over training iterations."""
    np.random.seed(42)
    x = np.random.randn(20, 8)
    y = np.random.randn(20, 1)
    params = init_mlp_weights(input_dim=8, seed=42)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    initial_loss = compute_mse_loss(forward_mlp(x, params)[0], y)
    for step in range(1, 25):
        _, cache = forward_mlp(x, params)
        grads = backward_mlp(y, cache, params)
        step_adam(params, grads, m, v, t=step, lr=0.01)

    final_loss = compute_mse_loss(forward_mlp(x, params)[0], y)
    assert final_loss < initial_loss
