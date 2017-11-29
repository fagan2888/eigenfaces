# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# time module
import time

if __name__ == '__main__':

    # argument parser instance
    parser = argparse.ArgumentParser()
    # init log level argument
    parser.add_argument('--log', type=str,
                        help='<optionalLog Level (info | debug)')
    # parse arguments
    argv = parser.parse_args()
    # get log level
    _level = argv.log or ''

    logger = logging.getLogger('app_b')

    if _level.upper() == 'INFO':
        coloredlogs.install(level='IFNO', logger=logger)
    elif _level.upper() == 'DEBUG':
        coloredlogs.install(level='DEBUG', logger=logger)
    else:
        coloredlogs.install(level='WARNING', logger=logger)

    logger.info('Fetching data...')
    data = fetch_data(ratio=0.8)

    X_train, y_train = data['train']

    D, N = X_train.shape
    logger.debug('Number of features: D=%d' % D)
    logger.debug('Number of train data: N=%d' % N)

    # mean face
    mean_face = X_train.mean(axis=1).reshape(-1, 1)

    A = X_train - mean_face
    logger.debug('A.shape=%s' % (A.shape,))

    S = (1 / N) * np.dot(A.T, A)
    logger.debug('S.shape=%s' % (S.shape,))

    # Calculate eigenvalues `w` and eigenvectors `v`
    logger.info('Calculating eigenvalues and eigenvectors...')
    _l, _v = np.linalg.eig(S)

    # Indexes of eigenvalues, sorted by value
    logger.info('Sorting eigenvalues...')
    _indexes = np.argsort(_l)[::-1]

    # TODO
    # threshold w's
    logger.warning('TODO: threshold eigenvalues')

    # Sorted eigenvalues and eigenvectors
    l = _l[_indexes]
    logger.debug('l.shape=%s' % (l.shape,))
    v = _v[:, _indexes]
    logger.debug('v.shape=%s' % (v.shape,))

    M = 2

    V = v[:, :M]
    logger.debug('V.shape=%s' % (V.shape,))

    _U = np.dot(A, V)

    U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
    logger.debug('U.shape=%s' % (U.shape,))

    W = np.dot(U.T, A)
    logger.debug('W.shape=%s' % (W.shape,))

    X_test, y_test = data['test']
    I, K = X_test.shape
    assert I == D, logger.error(
        'Number of features of test and train data do not match!')
    logger.debug('Number of features: D=%d' % I)
    logger.debug('Number of test data: K=%d' % K)

    Phi = X_test - mean_face
    logger.debug('Phi.shape=%s' % (Phi.shape,))

    W_test = np.dot(U.T, Phi)
    logger.debug('W_test.shape=%s' % (W_test.shape,))

    classes = set(y_train.ravel())

    accs = []

    classifiers = []

    classifier = SVC(kernel='linear', C=1e1,
                     decision_function_shape='ovo')

    classifier.fit(W.T, y_train.ravel())

    y_hat = classifier.predict(W_test.T)

    print(np.sum(y_hat == y_test.ravel()) / K)


"""
    _params = {'C': np.logspace(-3, 0, 10)}

    for c in list(classes):
        # preprocess training labels
        l_train = -np.ones(y_train.T.shape)
        l_train[y_train.T == c] = 1
        # preprocess testing labels
        l_test = -np.ones(y_test.T.shape)
        l_test[y_test.T == c] = 1

        search = GridSearchCV(SVC(kernel='linear', probability=True),
                              param_grid=_params, cv=3, n_jobs=-1)

        search.fit(W.T, l_train.ravel())

        # SVC(kernel='linear', C=10000, probability=True)
        classifier = search.best_estimator_

        print(search.best_params_)

        #classifier.fit(W.T, l_train.ravel())

        classifiers.append(classifier)

        y_hat_train = classifier.predict(W.T)

        acc_train = np.sum(l_train.ravel() == y_hat_train) / N
        # print(acc_train)
        # print(search.best_params_['gamma'])
        assert acc_train == 1.0  # , search.best_params_

        scores = classifier.predict(W_test.T)

        acc_test = np.sum(l_test.ravel() == scores) / K
        # print(acc_test)

    truuuuues = 0

    DIFFS = 0

    for j in range(W_test.T.shape[0]):

        w_test = W_test.T[j]
        l_test = y_test.T[j]

        probs = []

        for c in range(len(classifiers)):

            pred = classifiers[c].predict_proba(w_test.reshape(1, -1))
            probs.append(pred[:, 1])

        probs = np.array(probs)

        true_index = np.argmax(probs)

        if true_index != 4 and list(classes)[true_index] != 5:

            DIFFS += 1

        if l_test == list(classes)[true_index]:

            truuuuues += 1

    print(truuuuues / K)
"""
