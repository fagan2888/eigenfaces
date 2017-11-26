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

# utility functions
from utils import progress

SHAPE = (46, 56)

data = fetch_data()

X_train, y_train = data['train']

D, N = X_train.shape

# mean face
mean_face = X_train.mean(axis=1).reshape(-1, 1)

A = X_train - mean_face

S = (1 / N) * np.dot(A.T, A)

# Calculate eigenvalues `l` and eigenvectors `v`
_l, _v = np.linalg.eig(S)

# Indexes of eigenvalues, sorted by value
_indexes = np.argsort(_l)[::-1]

# TODO
# threshold w's

# Sorted eigenvalues and eigenvectors
l = _l[_indexes]
v = _v[:, _indexes]

M = np.arange(1, N)

error = []

for j, m in enumerate(M):

    progress(j, len(M), status='Reconstruction for M=%d, out of %d.' %
             (m, len(M)))

    V = v[:, :m]

    _U = np.dot(A, V)

    U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)

    W = np.dot(U.T, A)

    A_hat = np.dot(U, W)

    error.append(np.mean(np.sum((A - A_hat)**2)))

with plt.xkcd():

    plt.plot(M, error)

plt.show()
