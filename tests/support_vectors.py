# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [12.0, 9.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from src.reader import fetch_data
# custom PCA transformer
from src.pca import PCA
# custom One-Versus-Rest SVM
from src.ovr import OVR
# custom One-Versus-One SVM
from src.ovo import OVO

SHAPE = (46, 56)

M = 121
standard = True

data = fetch_data(ratio=0.8)

X_train, y_train = data['train']

D, N = X_train.shape

pca = PCA(n_comps=M, standard=standard)

W_train = pca.fit(X_train)

X_test, y_test = data['test']
I, K = X_test.shape

W_test = pca.transform(X_test)

scores = []

params = {'C': 1, 'gamma': 2e-4, 'kernel': 'linear'}

ovr = OVR(**params)
ovr.fit(W_train, y_train)

n_idx = 4

np.random.seed(13)

fig, axes = plt.subplots(nrows=2, ncols=n_idx)

idx = np.array(list(ovr.classifiers.keys()))[
    np.random.choice(len(ovr.classifiers.keys()), n_idx)]

for l, i in zip(idx, range(n_idx)):

    w_hat = ovr.classifiers[l].support_vectors_[0]
    x_hat = pca.reconstruct(w_hat)

    axes[0, i].imshow(x_hat.reshape(SHAPE).T,
                      cmap=plt.get_cmap('gray'))
    axes[0, i].set_title('OVR\nSupport Vector for label %d\n' % l)

ovo = OVO(**params)
ovo.fit(W_train, y_train)

idx = np.array(list(ovo.classifiers.keys()))[
    np.random.choice(len(ovo.classifiers.keys()), n_idx)]

for l, i in zip(idx, range(n_idx)):

    w_hat = ovo.classifiers[tuple(l)].support_vectors_[0]
    x_hat = pca.reconstruct(w_hat)

    axes[1, i].imshow(x_hat.reshape(SHAPE).T,
                      cmap=plt.get_cmap('gray'))
    axes[1, i].set_title('OVO\nSupport Vector for pair %s\n' % (tuple(l),))

fig.tight_layout()
fig.savefig('data/out/support_vectors.pdf',
            format='pdf', dpi=300, transparent=True)
