"""
Tests for `backend/app.py`.

These hit the flask routes through the test client.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from backend.app import app as flask_app
from backend import app as app_module


# ---------- fixtures / helpers ----------

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def array_to_data_url(arr: np.ndarray) -> str:
    """Build a base64 png data url from a uint8 grayscale array."""
    image = Image.fromarray(arr.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def blank_canvas(size: int = 280) -> np.ndarray:
    """Matches the frontend convention: black background without ink drawn."""
    return np.zeros((size, size), dtype=np.uint8)


def draw_square(canvas: np.ndarray, top: int, left: int, side: int, value: int = 236) -> np.ndarray:
    """Whiteish stroke on the black canvas, default value matches #EDEDE3 chalk color."""
    arr = canvas.copy()
    arr[top:top + side, left:left + side] = value
    return arr


# ---------- /health ----------

class TestHealth:
    def test_returns_200_and_ok_status(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"

    def test_reports_layer_sizes(self, client):
        response = client.get("/health")
        data = response.get_json()

        assert "layer_sizes" in data
        assert data["layer_sizes"][0] == 784   # input layer must be 784 (28x28 flattened)
        assert data["layer_sizes"][-1] == 10   # output layer must be 10 (digits 0-9)


# ---------- / (index) ----------

class TestIndex:
    def test_serves_frontend_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert response.content_type.startswith("text/html")


# ---------- /predict: error handling ----------

class TestPredictErrorHandling:
    def test_no_body_returns_400(self, client):
        response = client.post("/predict")

        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_json_missing_image_field_returns_400(self, client):
        response = client.post("/predict", json={"not_image": "whatever"})

        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_malformed_base64_returns_400_not_500(self, client):
        response = client.post("/predict", json={"image": "not-valid-base64-or-png!!!"})

        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_empty_string_image_returns_400(self, client):
        response = client.post("/predict", json={"image": ""})

        assert response.status_code == 400


# ---------- /predict: happy path ----------

class TestPredictHappyPath:
    def test_valid_drawing_returns_full_response_shape(self, client):
        arr = draw_square(blank_canvas(280), top=100, left=100, side=80)
        data_url = array_to_data_url(arr)

        response = client.post("/predict", json={"image": data_url})
        data = response.get_json()

        assert response.status_code == 200
        assert "prediction" in data
        assert "probabilities" in data
        assert isinstance(data["prediction"], int)
        assert 0 <= data["prediction"] <= 9
        assert len(data["probabilities"]) == 10

    def test_probabilities_sum_to_one(self, client):
        arr = draw_square(blank_canvas(280), top=100, left=100, side=80)
        data_url = array_to_data_url(arr)

        response = client.post("/predict", json={"image": data_url})
        probabilities = response.get_json()["probabilities"]

        assert sum(probabilities) == pytest.approx(1.0, abs=1e-4)

    def test_prediction_matches_argmax_of_probabilities(self, client):
        arr = draw_square(blank_canvas(280), top=100, left=100, side=80)
        data_url = array_to_data_url(arr)

        response = client.post("/predict", json={"image": data_url})
        data = response.get_json()

        assert data["prediction"] == int(np.argmax(data["probabilities"]))

    def test_blank_canvas_does_not_crash(self, client):
        data_url = array_to_data_url(blank_canvas(280))

        response = client.post("/predict", json={"image": data_url})
        data = response.get_json()

        assert response.status_code == 200
        assert sum(data["probabilities"]) == pytest.approx(1.0, abs=1e-4)


# ---------- /predict: preprocessing convention regression guard ----------

class TestPredictInkConvention:
    def test_feeds_network_with_correct_ink_convention(self, client, monkeypatch):
        """
        Directly inspects the array passed to net.forward(). Background
        corners should be near 0, and the region we drew on should be
        near 1 after normalization as this is the MNIST convention
        (ink = high, background = 0) that the model was trained on.
        """
        captured = {}
        original_forward = app_module.net.forward

        def spy_forward(x):
            captured["x"] = x
            return original_forward(x)

        monkeypatch.setattr(app_module.net, "forward", spy_forward)

        arr = draw_square(blank_canvas(280), top=40, left=40, side=200)
        data_url = array_to_data_url(arr)

        client.post("/predict", json={"image": data_url})

        assert "x" in captured
        x = captured["x"].reshape(28, 28)

        corner_avg = np.mean([x[0, 0], x[0, 27], x[27, 0], x[27, 27]])
        center_avg = x[10:18, 10:18].mean()

        assert corner_avg < 0.15, (
            f"corner (background) pixels averaged {corner_avg:.3f}, expected near 0, "
            "input may be inverted"
        )
        assert center_avg > 0.5, (
            f"center (ink) pixels averaged {center_avg:.3f}, expected near 1, "
            "input may be inverted"
        )
