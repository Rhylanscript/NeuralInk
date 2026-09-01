"""
Tests for model/preprocess.py.

Covers each pipeline stage in isolation (decode, crop, center of mass math,
shift, centering) plus the full pipeline end to end.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from model.preprocess import (
    center_of_mass, center_on_mass, crop_to_content,
    decode_base64_image, preprocess_canvas_image,
    shift_manual,
)


# ---------- helpers ----------

def array_to_data_url(arr: np.ndarray) -> str:
    """
    build a base64 PNG data url from a uint8 grayscale array, so tests
    can specify pixels directly instead of handwriting base64 blobs.
    """
    image = Image.fromarray(arr.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def blank_canvas(size: int = 280) -> np.ndarray:
    """All white canvas (matches an untouched HTML canvas before inversion)."""
    return np.full((size, size), 255, dtype=np.uint8)


def draw_square(canvas: np.ndarray, top: int, left: int, side: int, value: int = 0) -> np.ndarray:
    """
    Draw a filled square of `value` onto a copy of `canvas`. Default value 0
    (black) simulates ink on a white canvas, pre inversion.
    """
    arr = canvas.copy()
    arr[top:top + side, left:left + side] = value
    return arr


# ---------- decode_base64_image ----------

class TestDecodeBase64Image:
    def test_round_trip_shape(self):
        arr = blank_canvas(50)
        data_url = array_to_data_url(arr)

        image = decode_base64_image(data_url)

        assert image.size == (50, 50)

    def test_round_trip_pixel_values(self):
        arr = draw_square(blank_canvas(50), top=10, left=10, side=5, value=0)
        data_url = array_to_data_url(arr)

        image = decode_base64_image(data_url)
        decoded = np.array(image)

        assert decoded[12, 12] == 0
        assert decoded[0, 0] == 255

    def test_handles_missing_data_url_prefix(self):
        arr = blank_canvas(20)
        image = Image.fromarray(arr, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        image = decode_base64_image(raw_b64)

        assert image.size == (20, 20)


# ---------- crop_to_content ----------

class TestCropToContent:
    def test_crops_to_drawn_region(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        arr[10:20, 15:25] = 255  # a 10x10 "stroke" block

        image = Image.fromarray(arr, mode="L")
        cropped = crop_to_content(image)

        assert cropped.size == (10, 10)

    def test_blank_canvas_does_not_crash(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        image = Image.fromarray(arr, mode="L")

        cropped = crop_to_content(image)

        assert cropped.size == (50, 50)

    def test_ignores_low_noise_pixels(self):
        arr = np.zeros((50, 50), dtype=np.uint8)
        arr[:] = 5
        image = Image.fromarray(arr, mode="L")

        cropped = crop_to_content(image)

        assert cropped.size == (50, 50)


# ---------- center_of_mass ----------

class TestCenterOfMass:
    def test_centered_square_gives_center_coordinates(self):
        arr = np.zeros((28, 28), dtype=np.float64)
        arr[10:18, 10:18] = 1.0

        cy, cx = center_of_mass(arr)

        assert cy == pytest.approx(13.5, abs=0.01)
        assert cx == pytest.approx(13.5, abs=0.01)

    def test_off_center_blob_gives_off_center_coordinates(self):
        arr = np.zeros((28, 28), dtype=np.float64)
        arr[0:4, 0:4] = 1.0

        cy, cx = center_of_mass(arr)

        assert cy < 14
        assert cx < 14

    def test_empty_array_returns_nan(self):
        arr = np.zeros((28, 28), dtype=np.float64)

        cy, cx = center_of_mass(arr)

        assert np.isnan(cy)
        assert np.isnan(cx)

    def test_matches_hand_calculated_weighted_average(self):
        arr = np.zeros((3, 3), dtype=np.float64)
        arr[0, 0] = 1.0
        arr[2, 2] = 3.0

        cy, cx = center_of_mass(arr)

        assert cy == pytest.approx(1.5)
        assert cx == pytest.approx(1.5)


# ---------- shift_manual ----------

class TestShiftManual:
    def test_shift_moves_content_correctly(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        arr[2, 2] = 1.0

        shifted = shift_manual(arr, shift_y=3, shift_x=1)

        assert shifted[5, 3] == 1.0
        assert shifted.sum() == 1.0

    def test_negative_shift(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        arr[5, 5] = 1.0

        shifted = shift_manual(arr, shift_y=-2, shift_x=-2)

        assert shifted[3, 3] == 1.0

    def test_zero_shift_is_identity(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        arr[4, 4] = 1.0

        shifted = shift_manual(arr, shift_y=0, shift_x=0)

        assert np.array_equal(shifted, arr)

    def test_shift_off_edge_drops_content_without_crashing(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        arr[0, 0] = 1.0

        shifted = shift_manual(arr, shift_y=-5, shift_x=-5)

        assert shifted.sum() == 0.0
        assert shifted.shape == arr.shape


# ---------- center_on_mass ----------

class TestCenterOnMass:
    def test_off_center_input_ends_up_centered(self):
        small = np.zeros((20, 20), dtype=np.float64)
        small[0:4, 0:4] = 255.0

        result = center_on_mass(small, canvas_size=28)

        cy, cx = center_of_mass(result)
        assert cy == pytest.approx(14, abs=1.0)
        assert cx == pytest.approx(14, abs=1.0)

    def test_output_shape(self):
        small = np.zeros((20, 20), dtype=np.float64)
        small[8:12, 8:12] = 255.0

        result = center_on_mass(small, canvas_size=28)

        assert result.shape == (28, 28)

    def test_blank_input_does_not_crash(self):
        small = np.zeros((20, 20), dtype=np.float64)

        result = center_on_mass(small, canvas_size=28)

        assert result.shape == (28, 28)
        assert result.sum() == 0.0


# ---------- preprocess_canvas_image ----------

class TestPreprocessCanvasImageFullPipeline:
    def test_output_shape_and_range(self):
        arr = blank_canvas(280)
        arr = draw_square(arr, top=100, left=100, side=80, value=0)

        data_url = array_to_data_url(arr)
        result = preprocess_canvas_image(data_url)

        assert result.shape == (1, 784)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_blank_canvas_does_not_crash(self):
        arr = blank_canvas(280)

        data_url = array_to_data_url(arr)
        result = preprocess_canvas_image(data_url)

        assert result.shape == (1, 784)
        assert result.sum() == pytest.approx(0.0, abs=1e-6)

    def test_drawn_content_is_nonzero(self):
        arr = blank_canvas(280)
        arr = draw_square(arr, top=100, left=100, side=80, value=0)

        data_url = array_to_data_url(arr)
        result = preprocess_canvas_image(data_url)

        assert result.sum() > 0

    def test_small_and_large_strokes_both_produce_valid_output(self):
        small = draw_square(blank_canvas(280), top=140, left=140, side=5, value=0)
        large = draw_square(blank_canvas(280), top=10, left=10, side=260, value=0)

        for arr in (small, large):
            data_url = array_to_data_url(arr)
            result = preprocess_canvas_image(data_url)

            assert result.shape == (1, 784)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
