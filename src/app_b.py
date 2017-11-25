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

M = 250

V = v[:, :M]

_U = np.dot(A, V)

U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)

W = np.dot(U.T, A)

# train loop

# test loop

X_test, y_test = data['test']

error = 0

sz = X_test.shape[1]

for i in range(sz):

    progress(i, sz, status='Testing the %dth datapoint' % i)

    x = X_test[:, i].reshape(-1, 1)

    phi = (x - mean_face)

    w = np.dot(phi.T, U)

    E = np.mean((W - w.T)**2, axis=0)

    index = np.argmin(E)

    pred = y_train[:, index]
    targ = y_test[:, i]

    if pred != targ:
        error += 1

print(error / sz)
