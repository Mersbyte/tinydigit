"""Check ./tinydigit against the numpy model on all 10,000 MNIST test images
(P5 and P2, several maxval values), and check how it handles bad input.

With --malformed-only, run just the input-handling checks. They need no
MNIST download, which is what the sanitizer CI job uses."""
import hashlib
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import export
import mnist
import train

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "tinydigit")
SAMPLES = os.path.join(HERE, "samples")
# The committed model.npz. The exact accuracy below is only checked for it,
# because a model retrained with another BLAS library can differ slightly.
MODEL_SHA256 = "bd3b59e1d703c9e0fb3c3e91645150af64b5b102a66d7fbc0f250c924ad26068"
EXPECTED_CORRECT = 9778  # the test accuracy stated in README.md
SAMPLE_INDICES = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]  # first test image of digits 0-9
CHUNK = 500  # files per ./tinydigit call, well below any ARG_MAX
CONF_TOLERANCE = 0.051  # C prints the percentage with one decimal

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def tinydigit(*args):
    return subprocess.run([BINARY, *args], capture_output=True, text=True)


def run_c(paths):
    """Return (predictions, confidences), or an error string."""
    preds, confs = [], []
    for i in range(0, len(paths), CHUNK):
        chunk = paths[i:i + CHUNK]
        res = tinydigit(*chunk)
        if res.returncode != 0 or res.stderr:
            return f"exit {res.returncode}: {res.stderr.strip()[:200]}"
        lines = res.stdout.splitlines()
        if len(lines) != len(chunk):
            return f"{len(lines)} output lines for {len(chunk)} files"
        for path, line in zip(chunk, lines):
            prefix = f"{path}: " if len(chunk) > 1 else ""  # one file: no prefix
            m = re.fullmatch(r"(\d) \((\d{1,3}\.\d)%\)", line[len(prefix):])
            if not line.startswith(prefix) or not m:
                return f"unexpected output line {line!r}"
            preds.append(int(m.group(1)))
            confs.append(float(m.group(2)))
    return np.array(preds), np.array(confs)


def reference(params, x):
    """numpy predictions and confidences (in percent) for float32 inputs x."""
    logits = train.forward(params, x)
    p = np.exp(logits - logits.max(1, keepdims=True))
    return logits.argmax(1), 100 * (p / p.sum(1, keepdims=True)).max(1)


def p5(pixels, maxval=255):
    return f"P5\n28 28\n{maxval}\n".encode() + bytes(pixels.astype(np.uint8))


def p2(pixels, maxval=255):
    rows = (" ".join(str(int(v)) for v in pixels[r:r + 28]) for r in range(0, 784, 28))
    return f"P2\n# MNIST test image\n28 28\n{maxval}\n".encode() + "\n".join(rows).encode() + b"\n"


def compare(name, tmp, params, images, labels, encode, maxval=255):
    """Write every image with encode(), run C on them, compare with numpy.
    Returns the C predictions, or None if the C run failed."""
    q = np.round(images.astype(np.float64) * maxval / 255).astype(np.uint8)
    paths = []
    for i, img in enumerate(q):
        path = os.path.join(tmp, f"{name.replace(' ', '-')}-{i}.pgm")
        with open(path, "wb") as f:
            f.write(encode(img, maxval))
        paths.append(path)
    result = run_c(paths)
    if isinstance(result, str):
        check(f"{name}: C matches numpy", False, result)
        return None
    c_pred, c_conf = result
    py_pred, py_conf = reference(params, q.astype(np.float32) / np.float32(maxval))
    diff = int((c_pred != py_pred).sum())
    conf_diff = float(np.abs(c_conf - py_conf).max())
    check(f"{name}: C matches numpy", diff == 0 and conf_diff <= CONF_TOLERANCE,
          f"{diff} of {len(q)} predictions differ, max confidence difference "
          f"{conf_diff:.3f} points, C accuracy {(c_pred == labels).mean() * 100:.2f}%")
    return c_pred


def valid_inputs(px):
    """(name, file contents) that must classify like samples/7.pgm."""
    return [
        ("comment_after_width", b"P5\n28#c\n28 255\n" + px),
        ("comment_after_magic", b"P5#c\n28 28 255\n" + px),
        ("comment_after_maxval", b"P5\n28 28\n255#c\n" + px),
    ]


def bad_inputs(px):
    """(name, file contents, expected error) for malformed files."""
    hdr = b"P5\n28 28\n255\n"
    p2_body = " ".join(["0"] * 784).encode()
    return [
        ("not_pgm", b"hello", "not a PGM file"),
        ("p6", b"P6\n28 28\n255\n" + px * 3, "only P5 and P2 PGM files are supported"),
        ("no_space_after_magic", b"P528 28\n255\n" + px, "bad PGM header"),
        ("letters_in_header", b"P5\n28 2x\n255\n" + px, "bad PGM header"),
        ("vertical_tab_in_header", b"P5\v28 28\n255\n" + px, "bad PGM header"),
        ("crlf_header", b"P5\r\n28 28\r\n255\r\n" + px, "PGM header uses CRLF line endings"),
        ("space_then_comment_after_maxval", b"P5\n28 28\n255 # c\n" + px, "extra data after the image"),
        ("two_newlines_after_maxval", hdr + b"\n" + px, "extra data after the image"),
        ("truncated", hdr + px[:-1], "truncated or invalid pixel data"),
        ("trailing_byte", hdr + px + b"\n", "extra data after the image"),
        ("wrong_size", b"P5\n27 28\n255\n" + px[:756], "image must be 28x28"),
        ("maxval_0", b"P5\n28 28\n0\n" + px, "maxval must be between 1 and 255"),
        ("maxval_16bit", b"P5\n28 28\n65535\n" + px * 2, "maxval must be between 1 and 255"),
        ("p2_value_above_maxval", b"P2\n28 28\n15\n16 " + p2_body, "truncated or invalid pixel data"),
        ("p2_bad_token", b"P2\n28 28\n255\n12x " + p2_body, "truncated or invalid pixel data"),
        ("p2_extra_value", b"P2\n28 28\n255\n" + p2_body + b" 0\n", "extra data after the image"),
    ]


def malformed_checks(tmp):
    sample7 = os.path.join(SAMPLES, "7.pgm")
    with open(sample7, "rb") as f:
        px = f.read()[-784:]
    sample_out = tinydigit(sample7).stdout  # "7 (100.0%)\n" with the committed model

    for name, data in valid_inputs(px):
        path = os.path.join(tmp, name + ".pgm")
        with open(path, "wb") as f:
            f.write(data)
        res = tinydigit(path)
        check(f"valid input {name}", res.returncode == 0 and res.stdout == sample_out
              and res.stderr == "", f"exit {res.returncode}, {(res.stdout or res.stderr).strip()}")

    for name, data, expected in bad_inputs(px):
        path = os.path.join(tmp, name + ".pgm")
        with open(path, "wb") as f:
            f.write(data)
        res = tinydigit(path)
        got = res.stderr.strip()
        check(f"bad input {name}",
              res.returncode == 1 and res.stdout == "" and res.stderr == f"{path}: {expected}\n",
              f"exit {res.returncode}, {got[len(path) + 2:] if got.startswith(path) else got}")

    missing = os.path.join(tmp, "missing.pgm")
    res = tinydigit(sample7, missing)
    check("good and missing file together",
          res.returncode == 1 and res.stdout == f"{sample7}: {sample_out}"
          and res.stderr == f"{missing}: cannot open file\n", f"exit {res.returncode}")

    res = tinydigit()
    check("no arguments", res.returncode == 2 and res.stderr.startswith("usage: "),
          f"exit {res.returncode}")

    res = subprocess.run(["sh", "-c", '"$0" "$1" >&-', BINARY, sample7],
                         capture_output=True, text=True)
    check("closed stdout", res.returncode == 1 and res.stderr.endswith("write error\n"),
          f"exit {res.returncode}, {res.stderr.strip().rsplit(': ', 1)[-1]}")


def model_checks(tmp):
    with open(os.path.join(HERE, "model.npz"), "rb") as f:
        committed = hashlib.sha256(f.read()).hexdigest() == MODEL_SHA256
    params = dict(np.load(os.path.join(HERE, "model.npz")))
    with open(os.path.join(HERE, "weights.h")) as f:
        check("weights.h matches model.npz", f.read() == export.header(params))

    images, labels = mnist.load("t10k")
    py_pred, _ = reference(params, mnist.to_float(images))
    correct = int((py_pred == labels).sum())
    detail = f"{correct} of {len(labels)} = {correct / len(labels) * 100:.2f}%"
    if committed:
        check("numpy accuracy", correct == EXPECTED_CORRECT, f"{detail}, expected {EXPECTED_CORRECT}")
    else:
        print(f"warn numpy accuracy: {detail}; model.npz is not the committed model; "
              "skipping exact accuracy check")

    c_pred = compare("P5", tmp, params, images, labels, p5)
    if committed:
        correct = -1 if c_pred is None else int((c_pred == labels).sum())
        check("C accuracy", correct == EXPECTED_CORRECT, f"{correct} of {len(labels)}")
    compare("P2", tmp, params, images, labels, p2)
    for maxval in (1, 15, 100):
        compare(f"P5 maxval {maxval}", tmp, params, images, labels, p5, maxval)
        compare(f"P2 maxval {maxval}", tmp, params, images, labels, p2, maxval)

    ok, bad = True, []
    for d, i in enumerate(SAMPLE_INDICES):
        with open(os.path.join(SAMPLES, f"{d}.pgm"), "rb") as f:
            same = f.read() == b"P5\n28 28\n255\n" + bytes(images[i])
        first = labels[i] == d and not (labels[:i] == d).any()
        if not (same and first):
            ok, bad = False, bad + [f"{d}.pgm"]
    check("samples are the first test image of each digit", ok, ", ".join(bad))

    # A single file prints only "digit (confidence%)".
    ok = True
    for d, i in enumerate(SAMPLE_INDICES):
        res = tinydigit(os.path.join(SAMPLES, f"{d}.pgm"))
        m = re.fullmatch(r"(\d) \(\d{1,3}\.\d%\)\n", res.stdout)
        ok &= res.returncode == 0 and m is not None and int(m.group(1)) == py_pred[i]
    check("single-file output format", ok, tinydigit(os.path.join(SAMPLES, "7.pgm")).stdout.strip())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        if "--malformed-only" not in sys.argv[1:]:
            model_checks(tmp)
        malformed_checks(tmp)
    print("PASS" if not failures else f"FAIL ({len(failures)} checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
