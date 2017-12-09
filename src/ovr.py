# helper data preprocessor
from reader import fetch_data
from pca import PCA

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


import numpy as np
from sklearn.svm import SVC


class OVR(object):

    def __init__(self):
        # dictionary of L SVM classifiers
        self.classifiers = {}
        # list of label classes
        self.classes = []

    def fit(self, X_train, y_train):
        # L label classes
        self.classes = list(set(y_train.ravel()))
        # iterate over classes
        for l in self.classes:
            # binarise labels
            l_train = -np.ones(y_train.T.shape)
            # set labels to 1 if (class==l)
            l_train[y_train.T == l] = 1
            # init SVM binary classifier
            svm = SVC(kernel=params['kernel'],
                      C=params['C'],
                      gamma=params['gamma'],
                      probability=True)
            # fit SVM
            svm.fit(X_train.T, l_train.ravel())
            # store l^{th} trained SVM
            self.classifiers[l] = svm

    def predict(self, X_test):
        # probability scores
        scores = []
        # iterate over classes
        for l in self.classes:
            # get probability scores
            proba = self.classifiers[l].predict_proba(X_test.T)
            # get probability of class '1' only
            scores.append(proba[:, 1])
        # find probability maximizers
        maxes = np.argmax(np.array(scores), axis=0)
        # predictions
        y_hat = np.array(
            list(map(lambda l: self.classes[l], maxes))).reshape(1, -1)
        return y_hat

    def score(self, X_test, y_test):
        # get number of samples
        K = X_test.shape[1]
        # get predictions
        y_hat = self.predict(X_test)
        # get accuracy
        accuracy = np.sum(y_test == y_hat) / K
        return accuracy


ovr = OVR()
ovr.fit(W_train, y_train)
acc = ovr.score(W_test, y_test)
print('Accuracy = %.2f%%' % (acc * 100))
