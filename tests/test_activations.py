"""
Unit tests for `model/activations.py`

These highlight the exact mathematical behavior of the activation
functions, if a sign or shape gets flipped later during backprop
work, these should fail immediately and notify rather than 
producing a network that trains badly.
"""

import numpy as np

from model.activations import relu, relu_derivative, softmax


def test_relu_zeroes_negatives():
    Z = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32)
    result = relu(Z)
    expected = np.array([[0.0, 0.0, 0.0, 0.5, 2.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_relu_derivative_is_step_function():
    Z = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]], dtype=np.float32)
    result = relu_derivative(Z)
    # note: derivative at exactly 0 is conventionally treated as 0 here
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_softmax_rows_sum_to_one():
    Z = np.array([
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
    ], dtype=np.float32)
    result = softmax(Z)
    row_sums = np.sum(result, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-6)


def test_softmax_all_outputs_positive():
    Z = np.array([[-100.0, 0.0, 100.0]], dtype=np.float32)
    result = softmax(Z)
    # exterme logit gaps underflow to 0.0 in float32 so >= is used
    # rather than >
    assert np.all(result >= 0)


def test_softmax_moderate_inputs_are_strictly_positive():
    Z = np.array([[-5.0, 0.0, 5.0]], dtype=np.float32)
    result = softmax(Z)
    assert np.all(result > 0)


def test_softmax_numerical_stability_large_inputs():
    Z = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    result = softmax(Z)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    np.testing.assert_allclose(np.sum(result), 1.0, rtol=1e-6)


def test_softmax_highest_logit_gets_highest_probability():
    Z = np.array([[1.0, 5.0, 2.0]], dtype=np.float32)
    result = softmax(Z)
    assert np.argmax(result) == 1
