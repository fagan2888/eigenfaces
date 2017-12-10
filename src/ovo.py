# scientific computing library
import numpy as np
from sklearn.svm import SVC
import itertools


class OVO(object):

    def __init__(self, kernel='rbf', C=1, gamma=1e-5, degree=2):
        # dictionary of L(L-1)/2 SVM classifiers
        self.classifiers = {}
        # list of label pair combinations
        self.combinations = []
        # hyperparameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree

    def fit(self, X_train, y_train):
        # L label classes
        classes = list(set(y_train.ravel()))
        # L(L-1)/2 pair combinations of labels
        self.combinations = list(
            itertools.combinations(classes, 2))
        # iterate over combinations
        for c1, c2 in self.combinations:
            # binarise labels
            y = np.empty(y_train.T.shape)
            y[:] = np.nan
            y[y_train.T == c1] = 1
            y[y_train.T == c2] = 0
            # indeces of interest
            _index = ~np.isnan(y).ravel()
            # select the training examples
            X = X_train.T[_index]
            # select the label examples
            y = y[_index].ravel()
            # init SVM binary classifier
            svm = SVC(kernel=self.kernel,
                      C=self.C,
                      gamma=self.gamma,
                      degree=self.degree)
            # fit SVM
            svm.fit(X, y)
            # store (c1, c2) trained SVM
            self.classifiers[(c1, c2)] = svm

    def predict(self, X_test):
        # votes leaderboard
        votes = np.zeros(
            (len(self.combinations) + 1, X_test.shape[1]))
        # iterate over combinations
        for c1, c2 in self.combinations:
            # get votes
            preds = self.classifiers[(c1, c2)].predict(X_test.T)

            for j, s in enumerate(preds):
                c = c1 if s == 1 else c2
                votes[c, j] += 1
        # choose the class with most votes
        y_hat = np.argmax(votes, axis=0)
        return y_hat

    def score(self, X_test, y_test):
        # get number of samples
        K = X_test.shape[1]
        # get predictions
        y_hat = self.predict(X_test)
        # get accuracy
        accuracy = np.sum(y_test == y_hat) / K
        return accuracy
