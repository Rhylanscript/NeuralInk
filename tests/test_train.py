"""
Tests for `model/train.py` primarily a numerical gradient check, which
validates that the backpropagation math is correct
"""

import numpy as np

from model.losses import cross_entropy_loss
from model.network import NeuralNetwork
from model.train import backward


def test_gradient_check_matches_analytical_backprop():
    """
    Compares our analytical `backward()` gradients against numerically
    estimated gradients (finite differences) for a small sample of
    weights across every layer.
    """
    # small network + tiny batch keeps this test fast
    net = NeuralNetwork([10, 6, 4], seed=1)

    for i in range(net.num_layers):
        net.weights[i] = net.weights[i].astype(np.float64)
        net.biases[i] = net.biases[i].astype(np.float64)

    rng = np.random.default_rng(0)
    X = rng.random((5, 10))
    y_true = np.zeros((5, 4))
    y_true[np.arange(5), rng.integers(0, 4, size=5)] = 1.0  # random one-hot labels

    # --- analytical gradients (what backward() computes) ---
    activations, z_values = net.forward(X)
    grad_weights, _ = backward(net, activations, z_values, y_true)

    # --- numerical gradient check on a couple of weight entries ---
    epsilon = 1e-5
    tolerance = 1e-4

    def loss_for_current_weights():
        activations, _ = net.forward(X)
        return cross_entropy_loss(activations[-1], y_true)

    checked_count = 0

    for layer in range(net.num_layers):
        W = net.weights[layer]
        # checking every single weight would be slow so a handful per
        # layer is enough to catch a broken gradient formula
        rows_to_check = min(3, W.shape[0])
        cols_to_check = min(3, W.shape[1])

        for r in range(rows_to_check):
            for c in range(cols_to_check):
                original_value = W[r, c]

                W[r, c] = original_value + epsilon
                loss_plus = loss_for_current_weights()

                W[r, c] = original_value - epsilon
                loss_minus = loss_for_current_weights()

                W[r, c] = original_value  # restore

                numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
                analytical_grad = grad_weights[layer][r, c]

                diff = abs(numerical_grad - analytical_grad)
                assert diff < tolerance, (
                    f"Gradient mismatch at layer {layer}, W[{r},{c}]: "
                    f"analytical={analytical_grad:.8f}, numerical={numerical_grad:.8f}, "
                    f"diff={diff:.8f}"
                )
                checked_count += 1

    assert checked_count > 0


def test_loss_decreases_after_one_gradient_step():
    """
    A more lightweight regression test: confirms that taking a single
    gradient descent step reduces loss on same batch. doesnt validate 
    the math as well as the gradient check above, but is fast and good 
    for catching regressions later.
    """
    from model.train import update_weights

    net = NeuralNetwork([20, 10, 4], seed=2)

    rng = np.random.default_rng(1)
    X = rng.random((8, 20)).astype(np.float32)
    y_true = np.zeros((8, 4), dtype=np.float32)
    y_true[np.arange(8), rng.integers(0, 4, size=8)] = 1.0

    activations, z_values = net.forward(X)
    loss_before = cross_entropy_loss(activations[-1], y_true)

    grad_weights, grad_biases = backward(net, activations, z_values, y_true)
    update_weights(net, grad_weights, grad_biases, learning_rate=0.5)

    activations_after, _ = net.forward(X)
    loss_after = cross_entropy_loss(activations_after[-1], y_true)

    assert loss_after < loss_before
