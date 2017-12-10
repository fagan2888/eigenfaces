import numpy as np
from sklearn.svm import SVC


class OVR(object):

    def __init__(self, kernel='rbf', C=1, gamma=1e-5, degree=2):
        # dictionary of L SVM classifiers
        self.classifiers = {}
        # list of label classes
        self.classes = []
        # hyperparameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree

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
            svm = SVC(kernel=self.kernel,
                      C=self.C,
                      gamma=self.gamma,
                      degree=self.degree,
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
