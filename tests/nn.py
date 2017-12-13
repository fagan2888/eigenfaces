# helper data preprocessor
from src.reader import fetch_data
# custom PCA transformer
from src.pca import PCA
# KNN Classifer
from sklearn.neighbors import KNeighborsClassifier

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
acc = nn.score(W_test.T, y_test.T.ravel())
print('Accuracy = %.2f%%' % (acc * 100))
