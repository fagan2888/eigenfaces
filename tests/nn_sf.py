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
_palette = sns.color_palette("muted")
sns.set_style("ticks")

# helper data preprocessor
from src.reader import fetch_data
# custom PCA transformer
from src.pca import PCA
# KNN Classifer
from sklearn.neighbors import KNeighborsClassifier

SHAPE = (46, 56)

M = 121
standard = False

data = fetch_data(ratio=0.8)

X_train, y_train = data['train']

D, N = X_train.shape

pca = PCA(n_comps=M, standard=standard)

W_train = pca.fit(X_train)

X_test, y_test = data['test']
I, K = X_test.shape

W_test = pca.transform(X_test)

nn = KNeighborsClassifier(n_neighbors=1)
nn.fit(W_train.T, y_train.T.ravel())

y_hat = nn.predict(W_test.T)

done = {'success': False, 'failure': False}

fig, axes = plt.subplots(ncols=2)

for y, t, w in zip(y_hat, y_test.T.ravel(), W_test.T):
    if y == t and done['success'] is False:
        x_hat = pca.reconstruct(w)
        axes[0].imshow(x_hat.reshape(SHAPE).T,
                       cmap=plt.get_cmap('gray'))
        axes[0].set_title(
            'Successful NN Classification\nPredicted Class: %d, True Class: %d' % (y, t), color=_palette[1])
        done['success'] = True
    elif y != t and done['failure'] is False:
        x_hat = pca.reconstruct(w)
        axes[1].imshow(x_hat.reshape(SHAPE).T,
                       cmap=plt.get_cmap('gray'))
        axes[1].set_title(
            'Failed NN Classification\nPredicted Class: %d, True Class: %d' % (y, t), color=_palette[2])
        done['failure'] = True
    elif done['failure'] is True and done['success'] is True:
        break

fig.tight_layout()
fig.savefig('data/out/nn_class_images.pdf',
            format='pdf', dpi=300, transparent=True)
