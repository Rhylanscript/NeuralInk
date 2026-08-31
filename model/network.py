"""
Defines the NeuralNetwork class: a configurable, fully connected
feedforward network built on top of numpy.

Example:
    net = NeuralNetwork([784, 128, 10])
    # -> 784 input features, one hidden layer of 128 neurons (ReLU),
    #    10 output neurons (Softmax)
"""

import numpy as np

from model.activations import relu, softmax


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], seed: int | None = 42):
        """
        `layer_sizes`: e.g. `[784, 128, 10]` - first entry is the input
        size, last entry is the output size, everything in between is
        a hidden layer.

        `seed`: fixes numpys random generator so weight initialization
        is reproducible between runs. Set to `None` for true randomness.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least an input and output layer")

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]

            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out), dtype=np.float32)

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray):
        """
        Runs a forward pass through the network.

        X: input batch, shape (batch_size, layer_sizes[0])

        Returns:
            activations: list of A values for each layer, INCLUDING the
                input itself as activations[0]. Length = num_layers + 1.
            z_values: list of raw weighted sum values (pre activation) for
                each layer. Length = num_layers.

        Both A and Z returned (not just the final prediction) because
        backpropagation needs every intermediate value to compute gradients
        layer by layer.
        """
        activations = [X]
        z_values = []

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            z_values.append(Z)

            if i == self.num_layers - 1:
                # final layer: softmax turns logits into class probabilities
                A = softmax(Z)
            else:
                # hidden layers: ReLU
                A = relu(Z)

            activations.append(A)

        return activations, z_values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the predicted digit (0-9) for each sample in X - the
        index of the highest probability in the final softmax output.
        """
        activations, _ = self.forward(X)
        final_output = activations[-1]
        return np.argmax(final_output, axis=1)


if __name__ == "__main__":
    # quick check when run directly
    net = NeuralNetwork([784, 128, 10])

    print(f"Layer sizes: {net.layer_sizes}")
    for i, (W, b) in enumerate(zip(net.weights, net.biases)):
        print(f" - Layer {i}: W shape {W.shape}, b shape {b.shape}")

    # fake a tiny batch of 4 random 'images' to test the forward pass shape
    dummy_X = np.random.rand(4, 784).astype(np.float32)
    activations, z_values = net.forward(dummy_X)

    print(f"\nForward pass with batch of {dummy_X.shape[0]} samples:")
    print(f" - Output shape: {activations[-1].shape}")  # should be (4, 10)
    print(f" - Output row sums (should be ~1.0 each): {np.sum(activations[-1], axis=1)}")

    predictions = net.predict(dummy_X)
    print(f" - Predicted digits (untrained, meaningless): {predictions}")
