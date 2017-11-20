import numpy as np
import matplotlib.pyplot as plt
from reader import fetch_data

SHAPE = (46, 56)

data = fetch_data()

X_train, y_train = data['train']

N = X_train.shape[0]

# mean face
mean_face = X_train.mean(axis=1).reshape(-1, 1)

A = X_train - mean_face

plt.imshow(A[:, 0].reshape(SHAPE).T)

S = (1 / N) * np.dot(A.T, A)

# Calculate eigenvalues `w` and eigenvectors `v`
_w, _v = np.linalg.eig(S)

# Indexes of eigenvalues, sorted by value
_indexes = np.argsort(np.abs(_w))[::-1]

# TODO
# threshold w's

# Sorted eigenvalues and eigenvectors
w = _w[_indexes]
v = _v[:, _indexes]

M = [10]

for m in M:
    V = v[:, :m]

    print("V.shape", V.shape)

    W = N * np.dot(V.T, S)

    print("W.shape", W.shape)

    U = np.dot(A, V)

    print("U.shape", U.shape)

    yo = np.dot(U, W)

    plt.figure()
    plt.imshow(yo[:, 0].reshape(SHAPE).T)
    plt.title(m)

plt.show()
