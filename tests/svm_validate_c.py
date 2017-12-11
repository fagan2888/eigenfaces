# helper data preprocessor
from src.reader import fetch_data
# custom PCA transformer
from src.pca import PCA
# custom One-Versus-Rest SVM
from src.ovr import OVR
# custom One-Versus-One SVM
from src.ovo import OVO

# scientific computing library
import numpy as np

# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')
# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
# prettify plots
plt.rcParams['figure.figsize'] = [12.0, 9.0]
sns.set_palette(sns.color_palette("muted"))
_palette = sns.color_palette("muted")
sns.set_style("ticks")

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

params = {'gamma': 2e-4, 'kernel': 'linear'}

fine = 5

# validate OVR
C_ovr = np.logspace(-5, 10, fine)
accuracy_ovr = []
for c in C_ovr:
    ovr = OVR(C=c, **params)
    ovr.fit(W_train, y_train)
    accuracy_ovr.append(ovr.score(W_test, y_test) * 100)

argmax_ovr = np.argmax(np.array(accuracy_ovr))

plt.semilogx(C_ovr, accuracy_ovr, color=_palette[0])
plt.axvline(C_ovr[argmax_ovr], color=_palette[1], lw=2)
plt.title('Cross Validation of $\mathbf{C}$ for OVR-SVM')
plt.xlabel('$\mathbf{C}$')
plt.ylabel('Recognition Accuracy [%]')
plt.savefig('data/out/c_validation_ovr.pdf', format='pdf', dpi=300)

# validate OVO
C_ovo = np.logspace(-5, 5, fine)
accuracy_ovo = []
for c in C_ovo:
    ovo = OVO(C=c, **params)
    ovo.fit(W_train, y_train)
    accuracy_ovo.append(ovo.score(W_test, y_test) * 100)

argmax_ovo = np.argmax(np.array(accuracy_ovo))

plt.figure()

plt.semilogx(C_ovo, accuracy_ovo, color=_palette[0])
plt.axvline(C_ovo[argmax_ovo], color=_palette[1], lw=2)
plt.title('Cross Validation of $\mathbf{C}$ for OVO-SVM')
plt.xlabel('$\mathbf{C}$')
plt.ylabel('Recognition Accuracy [%]')
plt.savefig('data/out/c_validation_ovo.pdf', format='pdf', dpi=300)
