# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [32.0, 24.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from reader import fetch_data
from pca import PCA
# visualization functions
from visualize import plot_confusion_matrix

# utility functions
from utils import progress

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# time module
import time

# built-in tools
import itertools
import os

if __name__ == '__main__':

    # argument parser instance
    parser = argparse.ArgumentParser()
    # init log level argument
    parser.add_argument('-l', '--log', type=str,
                        help='<optional> Log Level (info | debug)')
    parser.add_argument('-m', '--n_comps', type=int,
                        help='<optional> Number of principle components')
    parser.add_argument('-s', '--standard', action='store_true',
                        help='<optional> Standardize data')
    # parse arguments
    argv = parser.parse_args()
    # get log level
    _level = argv.log or ''
    # get number of principle components
    M = argv.n_comps or 50
    # get flag of standardization
    standard = argv.standard or False

    logger = logging.getLogger(os.path.basename(__file__).replace('.py', ''))

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

    pca = PCA(n_comps=M, logger=logger)

    # normalise data
    _W_train = pca.fit(X_train)
    _W_mean = np.mean(_W_train, axis=1)
    logger.debug('_W_mean.shape=%s' % (_W_mean.shape,))
    _W_std = np.std(_W_train, axis=1)
    logger.debug('_W_std.shape=%s' % (_W_std.shape,))

    W_train = ((_W_train.T - _W_mean) / _W_std).T
    logger.debug('W_train.shape=%s' % (_W_train.shape,))

    X_test, y_test = data['test']
    I, K = X_test.shape
    assert I == D, logger.error(
        'Number of features of test and train data do not match, %d != %d' % (D, I))

    _W_test = pca.transform(X_test)

    W_test = ((_W_test.T - _W_mean) / _W_std).T
    logger.debug('W_test.shape=%s' % (W_test.shape,))

    classes = set(y_train.ravel())

    C = len(classes)

    combs = itertools.combinations(classes, 2)

    #_params = {'kernel': ['linear'], 'C': [10]}

    LEADERBOARD = np.zeros((C + 1, K))

    for c1, c2 in combs:

        # preprocess training labels
        l_train = np.empty(y_train.T.shape)
        l_train[:] = np.nan
        l_train[y_train.T == c1] = 1
        l_train[y_train.T == c2] = 0
        _index = ~np.isnan(l_train).ravel()
        l_train = l_train[_index].ravel()

        # select the training examples
        w_train = W_train.T[_index]

        classifier = SVC(kernel='linear', C=1)

        classifier.fit(w_train, l_train)

        scores = classifier.predict(W_test.T)

        for j, s in enumerate(scores):
            c = c1 if s == 1 else c2
            LEADERBOARD[c, j] += 1

    y_hat = np.argmax(LEADERBOARD, axis=0)

    acc = np.sum(y_test == y_hat) / K

    logger.error('Accuracy = %.2f%%' % (acc * 100))

    cnf_matrix = confusion_matrix(
        y_test.ravel(), y_hat.ravel(), labels=list(classes))

    # Plot non-normalized confusion matrix
    plt.figure()
    logger.info('Plotting confusion matrices...')
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='SVM One versus One - Confusion Matrix',
                          cmap=plt.cm.Reds)
    plt.savefig('data/out/svm_ovo_cnf_matrix.pdf', format='pdf', dpi=300)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='SVM One versus One - Normalized Confusion Matrix',
                          cmap=plt.cm.Reds)
    plt.savefig('data/out/svm_ovo_cnf_matrix_norm.pdf', format='pdf', dpi=300)
    logger.info(
        'Exported at data/out/svm_ovo_cnf_matrix.pdf & data/out/svm_ovr_cnf_matrix_norm.pdf...')
