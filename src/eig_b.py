# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 15.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from reader import fetch_data

SHAPE = (46, 56)

data = fetch_data()

X_train, y_train = data['train']

D, N = X_train.shape

# mean face
mean_face = X_train.mean(axis=1).reshape(-1, 1)

plt.imshow(mean_face.reshape(SHAPE).T)
plt.savefig('data/out/mean_face_q1b.eps', format='eps', dpi=1000)

A = X_train - mean_face

S = (1 / N) * np.dot(A.T, A)

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
plt.savefig('data/out/eigenvalues_q1b.eps', format='eps', dpi=1000)
