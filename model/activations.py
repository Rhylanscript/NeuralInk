"""
Activation functions and their derivatives

Forward pass functions transform layer outputs into activations.
Derivative functions are used during backpropagation to compute
how much each layers input contributed to the final error.
"""

import numpy as np

def relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU: f(Z) = max(0, Z)

    Used for hidden layers as cheap to compute and avoids the 
    vanishing gradient problem that sigmoid and tanh have in
    deeper networks since its gradient is either exactly 0 or
    exactly 1
    """
    return np.maximum(0, Z)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU: 1 where Z > 0 else 0

    Used during backprop to determine whether gradient flows 
    backward through a given neuron (it only flows through neurons
    that were 'active' - had a positive input during the 
    forward pass)
    """
    return (Z > 0).astype(np.float32)

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax: converts raw output scores ('logits') into a 
    probability distribution over classes that sums to 1 for each 
    sample.

    Z has shape (batch_size, num_classes). Row max is subtracted
    before exponentiating purely for numerical stability as exp() 
    of a large number (e.g. 1000) overflows to inf in float32, but 
    shifting every value in the row down by its max produces the 
    exact same softmax output while keeping all exponents <= 0.
    """
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
