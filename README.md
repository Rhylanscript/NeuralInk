# NeuralInk

A handwriting digit recognizer built entirely from scratch. The neural network including forward propagation, backpropagation, and gradient descent, is implemented by hand in numpy, without PyTorch, TensorFlow, or Keras. A Flask backend serves a browser based drawing canvas that sends your handwriting to the network.

Draw a digit, watch the network read it.

## How it works

- **Network**: a configurable feedforward architecture (currently `[784, 128, 10]`), trained on MNIST, reaching **97.89% validation accuracy**
- **Math**: He initialized weights, ReLU hidden layers, softmax output, cross entropy loss, mini batch gradient descent, all implemented manually (thanks wikipedia) and verified with a numerical gradient check
- **Backend**: Flask reconstructs the trained network from saved weights at startup and exposes a `/predict` endpoint that accepts a canvas drawing and returns a full probability distribution over digits 0-9
- **Preprocessing**: canvas input is cropped to content, resized to fit a 20×20 box, and centered by center of mass, replicating MNISTs actual construction process, so drawings match the distribution the network was trained on
- **Frontend**: plain HTML/CSS/JS canvas, no framework, with live debounced predictions as you draw and animated probability bars

## Running locally (without Docker)

Requires Python 3.11

```bash
# activate venv and install deps
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# executed once: download MNIST and train the model
python -m scripts.download_mnist
python -m model.train

# start the server
python -m backend.app
```

Open `http://127.0.0.1:5000`.

> Run modules with `-m` from the project root (e.g. `python -m backend.app`), not as direct file paths, the internal imports (`model.network`, etc.) are root relative

## Running with Docker

Requires Docker Desktop (shocker)

```bash
docker compose up --build
```

Open `http://localhost:5000`. Live-reload is enabled for `backend/`, `frontend/`, `model/`, and `weights/` via bind mounts, so most changes don't require a rebuild, only changes to `Dockerfile` or `requirements.txt` do.

To stop:

```bash
docker compose down
```

## Quick start (no cloning required)

```bash
docker run -p 5000:5000 ghcr.io/rhylanscript/neuralink:latest
```

Then open `http://localhost:5000`.

## Running tests

```bash
pip install -r requirements-dev.txt

python -m pytest tests/ -v
```

Includes unit tests for activations, the networks forward pass, the training loop, the preprocessing pipeline, and the flask api layer.

## Training from scratch

```bash
python -m scripts.download_mnist    # downloads MNIST idx files into data/
python -m model.train               # trains and saves weights/trained_weights.npz
```

Default configuration: architecture `[784, 128, 10]`, batch size 64, learning rate 0.1, 20 epochs.

## Tech stack

- **Math/ML**:          numpy only, no ML frameworks
- **Backend**:          Flask, Pillow (canvas image decoding)
- **Frontend**:         HTML/CSS/JS
- **Testing**:          pytest
- **CI**:               GitHub Actions (test suite + docker build verification)
- **Containerization**: Docker, docker compose

## License

This repository is under an [MIT License][license]

<!-- LINKS -->
[license]: LICENSE
