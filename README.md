# MLP Library

A modular and extensible Multi-Layer Perceptron (MLP) library supporting dense layers, convolutional layers, pooling, and various optimizers — designed for performance and flexibility.

---

## Prerequisites

Before building, ensure that all git submodules are initialized and updated:

## Check submodules

```bash
git submodule update --init --recursive
```

## Downlaod assets

```bash
bash scripts/download-dataset.sh
```
## Compiling

```bash
bash scripts/make.sh
cd build/release
make -j
```

## Running 

```bash
cd build/release
./cli
```

## TODO List

```
- [x] MLP (Red densa)
- [x] CNN (Anomena)
- [x] Max Pooling (Anomena)
- [ ] Optimizers
  - [x] SGD
  - [x] SGD Momentum
  - [b] AdamA
  - [ ] Newton + BFGS
- [ ] Improvements
  - [b] Leverage dynamic updating
  - [x] Batch normalization
  - [ ] Residual connections
  - [x] L2 Ridge
- [ ] GPU implementation CNN
- [ ] Visualizer
```
