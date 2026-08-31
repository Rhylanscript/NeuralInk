"""
Downloads the raw MNIST dataset in IDX format into the `data/` dir.

MNIST is given as 4 gzipped bin files:
- Training images + labels (60k)
- Test images + labels (10k)

Run this once before training:

`python scripts/download_mnist.py`

"""

import gzip
import os
import shutil

import requests

# url for mnist dataset
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# scripts/ -> project root -> data/
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_file(filename: str) -> str:
    """Downloads a single `.gz` file into `DATA_DIR` if not already present"""
    gz_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(gz_path):
        print(f"Already have {filename}, skipping download.")
        return gz_path

    url = BASE_URL + filename
    print(f"Downloading {filename} ...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(gz_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return gz_path


def extract_file(gz_path: str) -> str:
    """Extracts a `.gz` file into `DATA_DIR`, returning the extracted path."""
    out_path = gz_path[:-3]  # strip ".gz"

    if os.path.exists(out_path):
        print(f"Already extracted {os.path.basename(out_path)}, skipping.")
        return out_path

    print(f"Extracting {os.path.basename(gz_path)} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"MNIST data directory: {DATA_DIR}\n")

    for filename in FILES:
        gz_path = download_file(filename)
        extract_file(gz_path)

    print("\nDone! Raw IDX files are ready in data/.")


if __name__ == "__main__": main()
