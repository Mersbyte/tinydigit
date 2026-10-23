# tinydigit

A small neural network (784-64-10, ReLU) trained on MNIST with numpy, exported to a C header and run by 183 lines of C99 that use only the standard library and never call `malloc`.

```
$ make
$ ./tinydigit samples/7.pgm
7 (100.0%)
```

With several files, each line is prefixed with its file name. The ten samples (the first image of each digit in the MNIST test set):

```
$ ./tinydigit samples/*.pgm
samples/0.pgm: 0 (100.0%)
samples/1.pgm: 1 (99.8%)
samples/2.pgm: 2 (100.0%)
samples/3.pgm: 3 (94.9%)
samples/4.pgm: 4 (100.0%)
samples/5.pgm: 5 (61.5%)
samples/6.pgm: 6 (100.0%)
samples/7.pgm: 7 (100.0%)
samples/8.pgm: 8 (95.1%)
samples/9.pgm: 9 (99.6%)
```

A file that cannot be read as a 28x28 PGM gives an error message on stderr and exit code 1, instead of a prediction.

## Results

| | |
|---|---|
| MNIST test accuracy | 97.78% (9,778 of 10,000) |
| C predictions equal to the numpy model | 10,000 of 10,000, for P5 and P2, each with maxval 1, 15, 100 and 255 |
| Parameters | 50,890 float32 |
| Weights | 203,560 bytes (198.8 KiB) of `const` data |
| Buffers in `tinydigit.c` | 3,136-byte static input buffer, 296 bytes of stack arrays for the layers; buffers that stdio allocates are not counted |
| `tinydigit.c` | 183 lines |
| `predict()` time | 37.8 to 38.1 µs per image over 10 runs, excluding file reading |
| 10,000 PGM files in one call | 0.74 to 0.96 s over 5 runs (the first run was the slowest), including file reading |

All timings were measured on an Apple M2 Pro (macOS 27.0.1), on one core, with Apple clang 21 and `-O2`. The binary is 232,264 bytes on that machine.

## How it works

- `train.py` trains the network with plain numpy: 15 epochs, mini-batch SGD (batch 64, momentum 0.9, learning rate 0.05 with cosine decay), seed 0. It writes `model.npz`.
- `export.py` writes the weights to `weights.h` as `static const float` arrays (`%.9g`, so every float32 round-trips exactly) and puts one test image per digit in `samples/`.
- `tinydigit.c` reads a PGM file, scales pixels to [0, 1], runs one forward pass and prints the argmax with its softmax probability. The code itself never calls `malloc`; stdio (`fopen`, `getc`, `printf`) may allocate its own buffers.
- `test_compare.py` checks that `./tinydigit` predicts what the numpy model predicts on all 10,000 test images, that the accuracy is 9,778 of 10,000, that `weights.h` matches `model.npz`, and that malformed files give the expected error and exit code.

## Build and test

```
make            # $(CC) $(CPPFLAGS) -O2 -std=c99 -Wall -Wextra -Werror -ffp-contract=off ... $(LDFLAGS) -lm
make WERROR=   # the same without -Werror, if a newer compiler adds a warning
make test       # needs Python 3.10+ and numpy; downloads the MNIST test set into data/
```

`make` never regenerates `weights.h`; it uses the committed file. `-ffp-contract=off` stops the compiler from fusing multiply-adds, which keeps the forward pass arithmetic the same across compilers and CPUs on targets where `FLT_EVAL_METHOD` is 0, such as x86-64 with SSE and ARM with NEON. On 32-bit x86 using the x87 FPU (`FLT_EVAL_METHOD` 2), intermediate results can be kept at higher precision and may differ. The softmax uses `expf` from the platform's libm, so the printed percentage could in principle differ in the last digit between platforms. CI builds and tests on Linux (gcc) and macOS (clang).

## Train

```
python3 -m pip install -r requirements.txt
make regen      # runs train.py then export.py: downloads MNIST into data/,
                # writes model.npz, weights.h and samples/
make test
```

Training took 1.87 to 1.96 s of CPU time over 3 runs, on the same Apple M2 Pro with Python 3.14.7 and numpy 2.5.4, plus the MNIST download the first time. Running `train.py` several times on that machine gave a byte-identical `model.npz` each time. Results on other machines may differ slightly because numpy's matrix multiply depends on the BLAS library. `model.npz` and `weights.h` in this repository are the ones the numbers above come from. `make test` checks the exact accuracy of 9,778 only when `model.npz` has the SHA-256 of the committed file; for a retrained model it prints the accuracy with a warning, and still requires the C and numpy predictions to match.

## Limitations

- Input must be a 28x28 grayscale PGM (P5 or P2, maxval up to 255), holding a single image with nothing after the pixel data. A P5 header with CRLF line endings is rejected. There is no resizing and no other image format.
- The network expects MNIST-style images: a light digit on a dark background, centred and scaled like MNIST. A photo of a dark digit on white paper will be misclassified unless it is preprocessed the same way.
- It is a plain two-layer perceptron with about 2.2% test error; a small convolutional network does noticeably better.
- The percentage is the softmax output, not a calibrated probability.
- Accuracy is printed on the test set after every epoch; there is no separate validation split, so the hyperparameters were not chosen on held-out data.
- The weights are float32 and take about 199 KiB.
- The program reads files with stdio. Running the network on a device without a file system would need a different way to get the pixels into `x`. It has not been run on such a device.

## Data

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) is by Yann LeCun, Corinna Cortes and Christopher J. C. Burges, and was built from NIST Special Database 3 and Special Database 1. It is available under the [Creative Commons Attribution-Share Alike 3.0 license (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/). `mnist.py` downloads it from the `ossci-datasets` mirror that PyTorch uses and checks the MD5 sums.

The images in `samples/` are ten images from the MNIST test set, converted to PGM. They are under CC BY-SA 3.0; see [samples/LICENSE](samples/LICENSE). The trained weights in `model.npz` and `weights.h` were learned from MNIST.

## License

The code is under the MIT license (see [LICENSE](LICENSE)). The MIT license covers the code only, not the MNIST data or the images in `samples/`.
