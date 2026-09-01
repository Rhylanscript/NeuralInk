"""
Flask server: loads the trained network once at startup, exposes
`/predict` for the frontend canvas, and serves the frontend files.
"""

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from model.network import NeuralNetwork
from model.preprocess import preprocess_canvas_image

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "trained_weights.npz"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


def load_trained_network(weights_path: Path) -> NeuralNetwork:
    """
    Reconstruct a NeuralNetwork from saved weights. `layer_sizes` 
    comes back as a numpy array from np.savez, so cast it back to 
    a plain list of ints.
    """
    data = np.load(weights_path)
    layer_sizes = data["layer_sizes"].tolist()

    net = NeuralNetwork(layer_sizes=layer_sizes)

    num_layers = len(layer_sizes) - 1
    net.weights = [data[f"W{i}"] for i in range(num_layers)]
    net.biases = [data[f"b{i}"] for i in range(num_layers)]

    return net


net = load_trained_network(WEIGHTS_PATH)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "layer_sizes": net.layer_sizes})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return jsonify({"error": "Expected JSON body with an 'image' field (base64 PNG)."}), 400

    try:
        x = preprocess_canvas_image(payload["image"])
    except Exception as exc:
        return jsonify({"error": f"Failed to preprocess image: {exc}"}), 400

    activations, _ = net.forward(x)
    probabilities = activations[-1][0]
    predicted_digit = int(np.argmax(probabilities))

    return jsonify({
        "prediction": predicted_digit,
        "probabilities": probabilities.tolist(),
    })


if __name__ == "__main__": app.run(debug=True, port=5000)
