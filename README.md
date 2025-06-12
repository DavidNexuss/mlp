# MLP library

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
[X] MLP (Red densa)
[X] CNN (Anomena)
[X] Max Pooling (Anomena)
[ ] Optimizers
    [X] SGD
    [X] SGD Momentum
    [B] AdamA
    [ ] Newton + BFGS
[ ] Improvements
    [B] Leverage dynamic updating
    [X] Batch normalization
    [ ] Residual connections
    [X] L2 Ridge
[ ] GPU implementation CNN
[ ] Visualizer
