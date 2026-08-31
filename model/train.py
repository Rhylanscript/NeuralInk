"""
Training loop: forward pass -> loss -> backpropagation -> gradient
descent weight update, which is repeated over mini batches and epochs.

This is where the neural network actually learns
"""

import numpy as np

from model.activations import relu_derivative
from model.losses import accuracy, cross_entropy_loss
from model.network import NeuralNetwork


def backward(net: NeuralNetwork, activations: list, z_values: list, y_true: np.ndarray) -> tuple[list, list]:
    """
    Computes gradients for every weight and bias in the network via
    backpropagation.

    activations, z_values: returned by `net.forward(X)`
    y_true: one hot labels for this batch, shape (batch_size, num_classes)

    Returns:
        grad_weights, grad_biases: lists matching net.weights/net.biases
        in structure, containing dL/dW and dL/db for each layer.
    """
    batch_size = y_true.shape[0]
    num_layers = net.num_layers

    grad_weights:   list[np.ndarray] = [None] * num_layers  # type: ignore[list-item]
    grad_biases:    list[np.ndarray] = [None] * num_layers  # type: ignore[list-item]

    # --- output layer ---
    # softmax + crossentropy simplifies to this form
    y_pred = activations[-1]
    dZ = y_pred - y_true  # shape (batch_size, num_classes)

    # --- Walk backward through every layer ---
    for layer in reversed(range(num_layers)):
        A_prev = activations[layer]  # activation feeding INTO this layer

        grad_weights[layer] = (A_prev.T @ dZ) / batch_size
        grad_biases[layer] = np.mean(dZ, axis=0, keepdims=True)

        if layer > 0:
            # send error back to the previous layers activation
            # and gate it through that layers ReLU derivative
            dA_prev = dZ @ net.weights[layer].T
            dZ = dA_prev * relu_derivative(z_values[layer - 1])

    return grad_weights, grad_biases


def update_weights(net: NeuralNetwork, grad_weights: list, grad_biases: list, learning_rate: float):
    """
    Gradient descent: move every weight / bias a small step in the 
    direction that reduces loss (opposite the gradient)
    """
    for i in range(net.num_layers):
        net.weights[i] -= learning_rate * grad_weights[i]
        net.biases[i] -= learning_rate * grad_biases[i]


def train(
    net: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.1,
) -> dict:
    """
    Trains `net` in place using mini batch gradient descent.

    Each epoch: shuffle the training data, split into mini batches,
    and for each batch run forward -> backward -> weight update.
    After each epoch, evaluate loss / accuracy on the validation set
    and print progress.

    Returns a history dict with per epoch train / val loss and 
    accuracy, useful for plotting a learning curve later.
    """
    num_samples = X_train.shape[0]
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    rng = np.random.default_rng(seed=0)

    for epoch in range(1, epochs + 1):
        # shuffle training data each epoch so mini batches differ every
        # pass - prevents the network from learning any accidental
        # ordering in the dataset
        permutation = rng.permutation(num_samples)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_losses = []

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            activations, z_values = net.forward(X_batch)
            batch_loss = cross_entropy_loss(activations[-1], y_batch)
            epoch_losses.append(batch_loss)

            grad_weights, grad_biases = backward(net, activations, z_values, y_batch)
            update_weights(net, grad_weights, grad_biases, learning_rate)

        # --- end of epoch : evaluate on train (approx, using last batches
        # average) and full validation set ---
        train_loss = float(np.mean(epoch_losses))

        train_activations, _ = net.forward(X_train)
        train_acc = accuracy(train_activations[-1], y_train)

        val_activations, _ = net.forward(X_val)
        val_loss = cross_entropy_loss(val_activations[-1], y_val)
        val_acc = accuracy(val_activations[-1], y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

    return history


if __name__ == "__main__":
    # full end to end training run: python -m model.train
    from model.data_loader import load_mnist

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    print("Initializing neural network...")
    net = NeuralNetwork([784, 128, 10])

    print("Training...\n")
    history = train(
        net,
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=64,
        learning_rate=0.1,
    )

    print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")

    # save trained weights so the api can load them without
    # retraining every time the server starts
    weights_path = "weights/trained_weights.npz"
    save_dict = {}
    for i, (W, b) in enumerate(zip(net.weights, net.biases)):
        save_dict[f"W{i}"] = W
        save_dict[f"b{i}"] = b
    np.savez(weights_path, layer_sizes=np.array(net.layer_sizes), **save_dict)
    print(f"Saved trained weights to {weights_path}")
