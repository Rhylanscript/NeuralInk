"""
Parses raw MNIST IDX binary files into numpy arrays.

IDX format reference:

Images (idx3 - 3D array: [num_images, rows, cols]):
    bytes 0-3   magic number        (2051)
    bytes 4-7   number of images    (big endian int32)
    bytes 8-11  number of rows      (big endian int32)
    bytes 12-15 number of columns   (big endian int32)
    bytes 16+   raw pixel bytes, unsigned 8 bit, row major

Labels (idx1 - 1D array: [num_labels]):
    bytes 0-3   magic number        (2049)
    bytes 4-7   number of labels    (big endian int32)
    bytes 8+    raw label bytes, unsigned 8 bit
"""

import os
import struct

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_images(filename: str) -> np.ndarray:
    """
    Reads an IDX3 image file and returns an array of shape
    (num_images, 784) with pixel values of [0, 1].

    Each 28x28 image is flattened into a 784 length vector because the
    network takes a flat input vector rather than a 2D grid.
    """
    path = os.path.join(DATA_DIR, filename)

    

    with open(path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Bad magic number for image file {filename}: got {magic}, expected 2051")

        buffer = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(buffer, dtype=np.uint8)

        images = images.reshape(num_images, num_rows * num_cols).astype(np.float32)
        images /= 255.0

    return images


def load_labels(filename: str) -> np.ndarray:
    """
    Reads an IDX1 label file and returns a 1D array of shape
    (num_labels,) containing integer digit labels 0-9.
    """
    path = os.path.join(DATA_DIR, filename)

    with open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Bad magic number for label file {filename}: got {magic}, expected 2049")

        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)

    return labels


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Converts integer labels (e.g. 7) into one hot vectors
    (e.g. [0,0,0,0,0,0,0,1,0,0]) needed for cross entropy loss
    against a softmax output layer.
    """
    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


def load_mnist():
    """
    Loads and returns the full MNIST dataset:
        (train_images, train_labels, test_images, test_labels)

    train_images:   (60000, 784)    float32, values in [0, 1]
    train_labels:   (60000, 10)     float32 one hot vectors
    test_images:    (10000, 784)    float32, values in [0, 1]
    test_labels:    (10000, 10)     float32 one hot vectors
    """
    train_images = load_images("train-images-idx3-ubyte")
    train_labels_raw = load_labels("train-labels-idx1-ubyte")

    test_images = load_images("t10k-images-idx3-ubyte")
    test_labels_raw = load_labels("t10k-labels-idx1-ubyte")

    train_labels = one_hot_encode(train_labels_raw)
    test_labels = one_hot_encode(test_labels_raw)

    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_mnist()

    print(f"train_images shape: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"train_labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"test_images shape:  {test_images.shape}, dtype: {test_images.dtype}")
    print(f"test_labels shape:  {test_labels.shape}, dtype: {test_labels.dtype}")

    print(f"\nPixel value range:      [{train_images.min()}, {train_images.max()}]")
    print(f"Example label (one hot):  {train_labels[0]}")
    print(f"Example label (digit):    {np.argmax(train_labels[0])}")
