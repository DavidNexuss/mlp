# MLP library

## Compiling

```bash
bash scripts/make.sh
cd build/release
make -j
```
## TODO List
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
[X] MLP
[X] CNN
[X] Max Pooling
[ ] GPU implementation CNN
[ ] Visualizer

Implementation details

Experiments
- xor  classifier
- basic autoencoder
- mnist classifier
    -MNIST CNN
    -MNIST CNN + Pooled

Future work
- Add full support for other type of layers using GPU.
- Add other type of optimzers that are non backpropagation based, rmsprop, reinforced learning and others for problems where backpropagation and gradient descent is not avaliable.
- Explore other optimizers for backrpopagation
- Improve the fine tuning algorithm for the hyper parameters of the backrpopagation optimizers and add other initialization strategies

Conclusion 

