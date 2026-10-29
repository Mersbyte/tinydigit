# Contributing

Thanks for helping.

## Build and test

```
make            # builds ./tinydigit
make test       # needs Python 3.10+ and numpy; downloads the MNIST test set into data/
```

`make test` must pass. It checks that `./tinydigit` predicts what the numpy model predicts on all 10,000 test images and that malformed files give the expected error.

## Rules

- `tinydigit.c` stays C99 with only the standard library, compiles with `-Wall -Wextra -Werror`, and never calls `malloc`.
- Don't change `model.npz` or `weights.h` unless the change is about training; if you retrain, say so in the pull request and update the numbers in the README.
- Numbers in the README (accuracy, timings, sizes) must match what the code does. Say how you measured new ones.
- Keep a pull request to one change.

Security problems go through private reporting; see [SECURITY.md](SECURITY.md).
