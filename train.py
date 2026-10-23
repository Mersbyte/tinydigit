"""Train a 784-64-10 ReLU MLP on MNIST with numpy and save model.npz."""
import os

import numpy as np

import mnist

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 0
HIDDEN = 64
EPOCHS = 15
BATCH = 64
LR = 0.05
MOMENTUM = 0.9


def forward(params, x):
    """Return logits. Shared with test_compare.py as the reference model."""
    h = np.maximum(x @ params["W1"].T + params["b1"], 0)
    return h @ params["W2"].T + params["b2"]


def main():
    rng = np.random.default_rng(SEED)
    x_train, y_train = mnist.load("train")
    x_test, y_test = mnist.load("t10k")
    x_train, x_test = mnist.to_float(x_train), mnist.to_float(x_test)
    n_in = x_train.shape[1]

    # He initialisation; weights stored as [out, in] to match the C layout.
    params = {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (HIDDEN, n_in)),
        "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, np.sqrt(2 / HIDDEN), (10, HIDDEN)),
        "b2": np.zeros(10),
    }
    params = {k: v.astype(np.float32) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}

    for epoch in range(EPOCHS):
        lr = np.float32(LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS)))
        order = rng.permutation(len(x_train))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            x, y = x_train[idx], y_train[idx]
            h = np.maximum(x @ params["W1"].T + params["b1"], 0)
            logits = h @ params["W2"].T + params["b2"]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(y)), y] -= 1  # d(cross-entropy)/d(logits)
            d2 = p / np.float32(len(y))
            dh = (d2 @ params["W2"]) * (h > 0)
            grads = {"W2": d2.T @ h, "b2": d2.sum(0),
                     "W1": dh.T @ x, "b1": dh.sum(0)}
            for k in params:
                velocity[k] = MOMENTUM * velocity[k] - lr * grads[k]
                params[k] += velocity[k]
        acc = (forward(params, x_test).argmax(1) == y_test).mean()
        print(f"epoch {epoch + 1:2d}  lr {lr:.4f}  test accuracy {acc:.4f}")

    np.savez(os.path.join(HERE, "model.npz"), **params)
    print(f"saved model.npz, test accuracy {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
