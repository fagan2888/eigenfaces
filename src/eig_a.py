import numpy as np
import matplotlib.pyplot as plt
from reader import fetch_data

SHAPE = (46, 56)

data = fetch_data()

X_train, y_train = data['train']

N = X_train.shape[0]

# mean face
mean_face = X_train.mean(axis=1).reshape(-1, 1)

# plt.imshow(mean_face.reshape(SHAPE).T)
# plt.savefig('data/out/mean_face_q1a.eps', format='eps', dpi=1000)
# plt.show()

A = X_train - mean_face

S = (1 / N) * np.dot(A, A.T)

# Calculate eigenvalues `w` and eigenvectors `v`
_w, _v = np.linalg.eig(S)

# Indexes of eigenvalues, sorted by value
_indexes = np.argsort(np.abs(_w))

# TODO
# threshold w's

# Sorted eigenvalues and eigenvectors
w = _w[_indexes]
v = _v[:, _indexes]

plt.bar(range(len(w)), np.abs(w[::-1]))
plt.savefig('data/out/eigenvalues_q1a.eps', format='eps', dpi=1000)
