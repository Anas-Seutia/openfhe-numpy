#!/usr/bin/env python3
"""Download MNIST test set and convert to binary format for C++ inference."""

import gzip
import struct
import urllib.request
import numpy as np
from pathlib import Path

MNIST_URLS = {
    "images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "openfhe_numpy" / "cpp" / "data" / "mnist"


def download(url: str, dest: str):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)


def load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download(MNIST_URLS["images"], "/tmp/t10k-images.gz")
    download(MNIST_URLS["labels"], "/tmp/t10k-labels.gz")

    images = load_images("/tmp/t10k-images.gz")
    labels = load_labels("/tmp/t10k-labels.gz")
    print(f"Loaded {len(images)} images, {len(labels)} labels")

    for i in range(len(images)):
        label = int(labels[i])
        bin_path = OUTPUT_DIR / f"mnist_{i}_label_{label}.bin"
        txt_path = OUTPUT_DIR / f"mnist_{i}_label_{label}.txt"

        # Save as 784 doubles (float64), matching LoadMNISTImage expectation
        img_doubles = images[i].astype(np.float64).flatten()
        img_doubles.tofile(str(bin_path))

        with open(txt_path, "w") as f:
            f.write(f"label: {label}\nshape: 28 28\n")

        if i % 1000 == 0:
            print(f"  Written {i}/{len(images)} ...")

    print(f"Done. {len(images)} samples written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
