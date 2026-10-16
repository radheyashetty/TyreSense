import numpy as np
import torch
import torch.nn as nn
from src.mlp_numpy import (
    init_mlp_weights,
    forward_mlp,
    compute_mse_loss,
    backward_mlp,
    check_gradients,
    step_sgd_momentum,
    step_adam,
)


def test_numerical_gradient_check():
    """EXP-1: Maximum relative gradient error must be < 1e-6."""
    np.random.seed(42)
    x = np.random.randn(8, 7)
    y = np.random.randn(8, 1)
    params = init_mlp_weights(seed=42)

    max_rel_error = check_gradients(x, y, params, epsilon=1e-5)
    print(f"Max relative gradient error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-6, f"Gradient check failed with error {max_rel_error}"


def test_numpy_vs_pytorch_loss_gap():
    """EXP-1: NumPy MLP loss matches PyTorch loss within 5% after training."""
    np.random.seed(42)
    x_np = np.random.randn(32, 7).astype(np.float32)
    y_np = np.random.randn(32, 1).astype(np.float32)
    params = init_mlp_weights(seed=42)

    # Convert initial weights to PyTorch model
    torch_model = nn.Sequential(
        nn.Linear(7, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    with torch.no_grad():
        torch_model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        torch_model[0].bias.copy_(torch.from_numpy(params["b1"].squeeze()))
        torch_model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        torch_model[2].bias.copy_(torch.from_numpy(params["b2"].squeeze()))
        torch_model[4].weight.copy_(torch.from_numpy(params["W3"].T))
        torch_model[4].bias.copy_(torch.from_numpy(params["b3"].squeeze()))

    # Train NumPy MLP with vanilla SGD for 10 steps
    lr = 0.01
    np_params = {k: v.copy() for k, v in params.items()}
    for _ in range(10):
        pred_np, cache = forward_mlp(x_np, np_params)
        grads = backward_mlp(y_np, cache, np_params)
        for k in np_params:
            np_params[k] -= lr * grads[k]

    # Train PyTorch model with SGD
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_pt = torch.from_numpy(x_np)
    y_pt = torch.from_numpy(y_np)

    for _ in range(10):
        optimizer.zero_grad()
        out = torch_model(x_pt)
        loss = loss_fn(out, y_pt)
        loss.backward()
        optimizer.step()

    final_np_loss = compute_mse_loss(forward_mlp(x_np, np_params)[0], y_np)
    final_pt_loss = float(loss_fn(torch_model(x_pt), y_pt).item())
    gap = abs(final_np_loss - final_pt_loss) / final_pt_loss

    print(f"NumPy loss: {final_np_loss:.6f}, PyTorch loss: {final_pt_loss:.6f}, Gap: {gap * 100:.2f}%")
    assert gap < 0.05, f"Loss gap between NumPy and PyTorch was {gap * 100:.2f}% (exceeds 5%)"


def test_adam_optimizer_decreases_loss():
    """Verify that Adam optimizer updates decrease loss over training iterations."""
    np.random.seed(42)
    x = np.random.randn(20, 7)
    y = np.random.randn(20, 1)
    params = init_mlp_weights(seed=42)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    initial_loss = compute_mse_loss(forward_mlp(x, params)[0], y)
    for step in range(1, 25):
        pred, cache = forward_mlp(x, params)
        grads = backward_mlp(y, cache, params)
        step_adam(params, grads, m, v, t=step, lr=0.01)

    final_loss = compute_mse_loss(forward_mlp(x, params)[0], y)
    assert final_loss < initial_loss
