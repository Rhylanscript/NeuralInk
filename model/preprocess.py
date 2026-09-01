"""
Converts a raw canvas drawing (base64 PNG) into a 784 dim vector
matching MNISTs preprocessing conventions:
  - white digit strokes on black background
  - digit resized to fit a 20x20 box, aspect ratio preserved
  - centered in a 28x28 frame via center of mass
  - normalized to [0, 1]
"""

import base64
import io

import numpy as np
from PIL import Image


def decode_base64_image(data_url: str) -> Image.Image:
    """
    Strip the 'data:image/png;base64,' prefix if present and decode
    into a PIL Image.
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")    # greyscale
    return image


def crop_to_content(image: Image.Image) -> Image.Image:
    """
    Crop the image to the bounding box of non background pixels.
    Assumes the image is already inverted so background = 0 (black)
    and strokes = high values.
    """
    arr = np.array(image)
    rows = np.any(arr > 10, axis=1)
    cols = np.any(arr > 10, axis=0)

    if not rows.any() or not cols.any():
        return image    # blank canvas

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    return image.crop((col_min, row_min, col_max + 1, row_max + 1))


def resize_to_20x20_box(image: Image.Image) -> Image.Image:
    """
    Resize so the longer side fits in 20px, preserving aspect ratio.
    Uses anti aliased (LANCZOS) resampling, matching MNISTs smooth
    grayscale edges rather than hard binary pixels.
    """
    width, height = image.size
    if width == 0 or height == 0:
        return image

    if width > height:
        new_width = 20
        new_height = max(1, round(height * (20 / width)))
    else:
        new_height = 20
        new_width = max(1, round(width * (20 / height)))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def center_of_mass(arr: np.ndarray) -> tuple[float, float]:
    """
    Weighted average of row/col indices, weighted by pixel intensity.
    """
    total = arr.sum()
    if total == 0:
        return (np.nan, np.nan)

    rows = np.arange(arr.shape[0])
    cols = np.arange(arr.shape[1])

    cy = float((rows[:, None] * arr).sum() / total)
    cx = float((cols[None, :] * arr).sum() / total)
    return cy, cx


def shift_manual(arr: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """
    Shift a 2d array by integer (shift_y, shift_x), filling
    vacated space with 0.
    """
    result = np.zeros_like(arr)
    h, w = arr.shape

    src_y_start = max(0, -shift_y)
    src_y_end = min(h, h - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(w, w - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[src_y_start:src_y_end, src_x_start:src_x_end]

    return result


def center_on_mass(arr: np.ndarray, canvas_size: int = 28) -> np.ndarray:
    frame = np.zeros((canvas_size, canvas_size), dtype=np.float64)

    h, w = arr.shape
    top = (canvas_size - h) // 2
    left = (canvas_size - w) // 2
    frame[top:top + h, left:left + w] = arr

    cy, cx = center_of_mass(frame)
    if np.isnan(cy) or np.isnan(cx):
        return frame

    shift_y = int(round(canvas_size / 2 - cy))
    shift_x = int(round(canvas_size / 2 - cx))

    return shift_manual(frame, shift_y, shift_x)

def preprocess_canvas_image(data_url: str) -> np.ndarray:
    """
    Full pipeline: base64 PNG -> normalized 784 dim vector ready
    for `NeuralNetwork.forward()`.
    """
    image = decode_base64_image(data_url)

    image = crop_to_content(image)
    image = resize_to_20x20_box(image)

    arr = np.array(image, dtype=np.float64)
    arr = center_on_mass(arr, canvas_size=28)

    arr = arr / 255.0
    return arr.flatten().reshape(1, -1)  # shape (1, 784) matches networks batch 
