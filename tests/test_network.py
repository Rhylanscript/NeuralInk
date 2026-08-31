"""
Unit tests for `model/network.py`

Focused on structural correctness (shapes, initialization). The network
is untrained right now, so accuracy isnt a meaningful thing to test yet.
"""

import numpy as np

from model.network import NeuralNetwork


def test_weight_and_bias_shapes():
    net = NeuralNetwork([784, 128, 10])

    assert net.weights[0].shape == (784, 128)
    assert net.biases[0].shape == (1, 128)

    assert net.weights[1].shape == (128, 10)
    assert net.biases[1].shape == (1, 10)


def test_forward_pass_output_shape():
    net = NeuralNetwork([784, 128, 10])
    X = np.random.rand(5, 784).astype(np.float32)

    activations, z_values = net.forward(X)

    assert activations[-1].shape == (5, 10)
    assert len(activations) == 3   # input + hidden + output
    assert len(z_values) == 2      # one per weight layer


def test_forward_pass_output_is_valid_probability_distribution():
    net = NeuralNetwork([784, 128, 10])
    X = np.random.rand(3, 784).astype(np.float32)

    activations, _ = net.forward(X)
    output = activations[-1]

    row_sums = np.sum(output, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], rtol=1e-6)
    assert np.all(output >= 0)


def test_same_seed_gives_identical_weights():
    net_a = NeuralNetwork([784, 128, 10], seed=42)
    net_b = NeuralNetwork([784, 128, 10], seed=42)

    np.testing.assert_array_equal(net_a.weights[0], net_b.weights[0])


def test_different_seed_gives_different_weights():
    net_a = NeuralNetwork([784, 128, 10], seed=1)
    net_b = NeuralNetwork([784, 128, 10], seed=2)

    assert not np.array_equal(net_a.weights[0], net_b.weights[0])


def test_predict_returns_valid_digit_range():
    net = NeuralNetwork([784, 128, 10])
    X = np.random.rand(10, 784).astype(np.float32)

    predictions = net.predict(X)

    assert predictions.shape == (10,)
    assert np.all(predictions >= 0) and np.all(predictions <= 9)


def test_supports_arbitrary_layer_depth():
    net = NeuralNetwork([784, 256, 128, 64, 10])

    assert len(net.weights) == 4
    assert net.weights[0].shape == (784, 256)
    assert net.weights[-1].shape == (64, 10)
