"""Download (once) and load MNIST as numpy arrays."""
import gzip
import hashlib
import os
import shutil
import urllib.request

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TIMEOUT = 60  # seconds without data before the download gives up


def _fetch(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {MIRROR + name}")
        with urllib.request.urlopen(MIRROR + name, timeout=TIMEOUT) as r, \
                open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    if digest != FILES[name]:
        raise SystemExit(f"{path}: md5 {digest}, expected {FILES[name]}")
    with gzip.open(path, "rb") as f:
        return f.read()


def _header(raw, count):
    return [int.from_bytes(raw[i:i + 4], "big") for i in range(0, 4 * count, 4)]


def _images(name):
    raw = _fetch(name)
    magic, n, rows, cols = _header(raw, 4)
    if magic != 2051 or rows != 28 or cols != 28 or len(raw) != 16 + n * 784:
        raise SystemExit(f"{name}: not a 28x28 IDX image file")
    return np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows * cols)


def _labels(name):
    raw = _fetch(name)
    magic, n = _header(raw, 2)
    if magic != 2049 or len(raw) != 8 + n:
        raise SystemExit(f"{name}: not an IDX label file")
    return np.frombuffer(raw, np.uint8, offset=8)


def load(split):
    """Return (images uint8 [n, 784], labels uint8 [n]) for 'train' or 't10k'."""
    images = _images(f"{split}-images-idx3-ubyte.gz")
    labels = _labels(f"{split}-labels-idx1-ubyte.gz")
    if len(images) != len(labels):
        raise SystemExit(f"{split}: {len(images)} images but {len(labels)} labels")
    return images, labels


def to_float(images):
    """uint8 pixels -> float32 in [0, 1], the same scaling the C code uses."""
    return images.astype(np.float32) / np.float32(255)
