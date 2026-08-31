"""
Loss function: categorical cross entropy.

Measures how far the networks predicted probability distribution is
from the true onehot label.
"""

import numpy as np


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    y_pred: softmax output, shape (batch_size, num_classes), each row sums to 1.
    y_true: one-hot labels, same shape.

    Formula per sample: `L = -sum(y_true * log(y_pred))`
    Since y_true is one hot, this collapses to -log(predicted probability
    assigned to the correct class). Mean loss across the batch is 
    returned so the value doesn't scale with batch size.

    A small epsilon is added inside the log to avoid `log(0) = -inf`,
    which would happen if the network ever predicts exactly 0 probability
    for the correct class (rare, but possible with float32 underflow as
    demonstrated by the softmax test.
    """
    epsilon = 1e-9
    batch_size = y_pred.shape[0]

    log_probs = np.log(y_pred + epsilon)
    loss = -np.sum(y_true * log_probs) / batch_size

    return float(loss)

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fraction of samples where the highest probability class matches
    the true label. Both inputs shape (batch_size, num_classes).
    """
    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return float(np.mean(predicted_classes == true_classes))
