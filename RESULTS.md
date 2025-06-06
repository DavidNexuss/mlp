CPU Results:

```
================[XOR TEST]========================
Epoch 0, Loss: 0.126838
Epoch 10, Loss: 0.0999698
Epoch 20, Loss: 0.0577204
Epoch 30, Loss: 0.0218449
Epoch 40, Loss: 0.0085146
Epoch 50, Loss: 0.00409764
Epoch 60, Loss: 0.00231695
Epoch 70, Loss: 0.00148032
Epoch 80, Loss: 0.00101463
Epoch 81, Loss: 0.000978553
Early stopping at epoch 81
Input: [0, 0] [0.0808627]
Input: [0, 1] [0.979265]
Input: [1, 0] [0.980829]
Input: [1, 1] [0.0181608]

================[AUTOENCODER]========================
Epoch 0, Loss: 0.366153
Epoch 10, Loss: 0.259878
Epoch 20, Loss: 0.0319872
Epoch 30, Loss: 0.0057525
Epoch 40, Loss: 0.0012466
Epoch 42, Loss: 0.000950839
Early stopping at epoch 42
Input: [0, 0, 1] [0.0676271, 0.035535, 0.967138]
Input: [0, 1, 1] [7.36989e-05, 1, 0.998536]
Input: [1, 0, 1] [0.999979, 2.97855e-05, 0.998623]
Input: [1, 1, 0] [0.999305, 0.999599, 0.0019015]

================[MNIST TEST]========================
Reading large dataset...
tag count = 10
1 # filecount = 5949 w: 28 h: 28 c: 1
2 # filecount = 5842 w: 28 h: 28 c: 1
3 # filecount = 5918 w: 28 h: 28 c: 1
4 # filecount = 6742 w: 28 h: 28 c: 1
5 # filecount = 6131 w: 28 h: 28 c: 1
6 # filecount = 5958 w: 28 h: 28 c: 1
7 # filecount = 5851 w: 28 h: 28 c: 1
8 # filecount = 5923 w: 28 h: 28 c: 1
9 # filecount = 5421 w: 28 h: 28 c: 1
10 # filecount = 6265 w: 28 h: 28 c: 1
Shuffling
Done
Reading large dataset...
tag count = 10
1 # filecount = 1009 w: 28 h: 28 c: 1
2 # filecount = 982 w: 28 h: 28 c: 1
3 # filecount = 958 w: 28 h: 28 c: 1
4 # filecount = 1135 w: 28 h: 28 c: 1
5 # filecount = 1010 w: 28 h: 28 c: 1
6 # filecount = 1032 w: 28 h: 28 c: 1
7 # filecount = 974 w: 28 h: 28 c: 1
8 # filecount = 980 w: 28 h: 28 c: 1
9 # filecount = 892 w: 28 h: 28 c: 1
10 # filecount = 1028 w: 28 h: 28 c: 1
Shuffling
Done
====(DNN)===
Epoch 0, Loss: 0.289774
Epoch 6, Loss: 0.000349296
Early stopping at epoch 6
Performing testing...  loss = 0.103883

====(CNN)===
Epoch 0, Loss: 0.204302
 Epoch 4, Loss: 0.000530185
Early stopping at epoch 4
Performing testing...  loss = 0.0571953

```
