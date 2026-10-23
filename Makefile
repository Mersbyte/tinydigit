CC ?= cc
CFLAGS ?= -O2
# Turn off with `make WERROR=` if a newer compiler adds a warning.
WERROR ?= -Werror
# Always added, even with `make CFLAGS=...`. -ffp-contract=off keeps the
# compiler from fusing a*b+c into FMA instructions.
override CFLAGS += -std=c99 -Wall -Wextra $(WERROR) -ffp-contract=off
PYTHON ?= python3

tinydigit: tinydigit.c weights.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ tinydigit.c $(LDFLAGS) -lm

test: tinydigit
	$(PYTHON) test_compare.py

# Retrain and re-export model.npz, weights.h and samples/. Never runs on its
# own; needs numpy and downloads MNIST into data/.
regen:
	$(PYTHON) train.py
	$(PYTHON) export.py

clean:
	rm -rf tinydigit tinydigit.dSYM

.PHONY: test regen clean
