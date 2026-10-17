import numpy as np


def relu(x):
    """Rectified Linear Unit activation function."""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Derivative of ReLU activation function."""
    return (x > 0.0).astype(float)


def init_mlp_weights(input_dim=8, seed=42):
    """Initialize weights and biases for an MLP using He initialization."""
    np.random.seed(seed)
    params = {
        "W1": np.random.randn(input_dim, 16) * np.sqrt(2.0 / input_dim),
        "b1": np.zeros((1, 16)),
        "W2": np.random.randn(16, 8) * np.sqrt(2.0 / 16),
        "b2": np.zeros((1, 8)),
        "W3": np.random.randn(8, 1) * np.sqrt(2.0 / 8),
        "b3": np.zeros((1, 1)),
    }
    return params


def forward_mlp(x, params):
    """Compute forward pass and cache activations for backpropagation."""
    z1 = np.dot(x, params["W1"]) + params["b1"]
    a1 = relu(z1)
    z2 = np.dot(a1, params["W2"]) + params["b2"]
    a2 = relu(z2)
    z3 = np.dot(a2, params["W3"]) + params["b3"]
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
    return z3, cache


def compute_mse_loss(y_pred, y_true):
    """Compute mean squared error loss between predictions and targets."""
    return float(np.mean((y_pred - y_true) ** 2))


def backward_mlp(y_true, cache, params):
    """Compute analytical gradients of MSE loss with respect to all parameters."""
    num_samples = len(y_true)
    dz3 = (2.0 / num_samples) * (cache["z3"] - y_true)
    grad_w3 = np.dot(cache["a2"].T, dz3)
    grad_b3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = np.dot(dz3, params["W3"].T)
    dz2 = da2 * relu_derivative(cache["z2"])
    grad_w2 = np.dot(cache["a1"].T, dz2)
    grad_b2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, params["W2"].T)
    dz1 = da1 * relu_derivative(cache["z1"])
    grad_w1 = np.dot(cache["x"].T, dz1)
    grad_b1 = np.sum(dz1, axis=0, keepdims=True)

    return {
        "W1": grad_w1, "b1": grad_b1,
        "W2": grad_w2, "b2": grad_b2,
        "W3": grad_w3, "b3": grad_b3
    }


def compute_numerical_gradient(x, y_true, params, param_name, epsilon=1e-5):
    """Compute numerical gradient for a specific parameter using finite differences."""
    grad = np.zeros_like(params[param_name])
    it = np.nditer(params[param_name], flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig_val = params[param_name][idx]

        params[param_name][idx] = orig_val + epsilon
        loss_plus = compute_mse_loss(forward_mlp(x, params)[0], y_true)

        params[param_name][idx] = orig_val - epsilon
        loss_minus = compute_mse_loss(forward_mlp(x, params)[0], y_true)

        params[param_name][idx] = orig_val
        grad[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
    return grad


def check_gradients(x, y_true, params, epsilon=1e-5):
    """Verify analytical gradients against numerical gradients across all layers."""
    _, cache = forward_mlp(x, params)
    analytical_grads = backward_mlp(y_true, cache, params)
    max_relative_error = 0.0

    for key in params:
        num_grad = compute_numerical_gradient(x, y_true, params, key, epsilon)
        ana_grad = analytical_grads[key]
        diff = np.abs(num_grad - ana_grad)
        norm = np.maximum(np.abs(num_grad), np.abs(ana_grad)) + 1e-8
        rel_error = float(np.max(diff / norm))
        if rel_error > max_relative_error:
            max_relative_error = rel_error
    return max_relative_error


def step_sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    """Update parameters using SGD with momentum."""
    for key in params:
        velocity[key] = momentum * velocity[key] + grads[key]
        params[key] = params[key] - lr * velocity[key]


def step_adam(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Update parameters using Adam optimizer."""
    for key in params:
        m[key] = beta1 * m[key] + (1.0 - beta1) * grads[key]
        v[key] = beta2 * v[key] + (1.0 - beta2) * (grads[key] ** 2)
        m_hat = m[key] / (1.0 - beta1 ** t)
        v_hat = v[key] / (1.0 - beta2 ** t)
        params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
