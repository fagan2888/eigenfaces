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

# utility functions
from utils import progress

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# time module
import time

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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

    M = 100

    pca = PCA(n_comps=M, logger=logger)

    W_train = pca.fit(X_train)
    logger.debug('W_train.shape=%s' % (W_train.shape,))

    X_test, y_test = data['test']
    I, K = X_test.shape
    assert I == D, logger.error(
        'Number of features of test and train data do not match!')

    W_test = pca.transform(X_test)
    logger.debug('W_test.shape=%s' % (W_test.shape,))

    classes = set(y_train.ravel())

    #_params = {'kernel': ['linear'], 'C': [10]}

    yo = []

    for c in classes:

        # preprocess training labels
        l_train = -np.ones(y_train.T.shape)
        l_train[y_train.T == c] = 1
        # preprocess testing labels
        l_test = -np.ones(y_test.T.shape)
        l_test[y_test.T == c] = 1

        # search.best_estimator_
        classifier = SVC(kernel='linear', C=10, probability=True)

        classifier.fit(W_train.T, l_train.ravel())

        y_hat_train = classifier.predict(W_train.T)

        acc_train = np.sum(l_train.ravel() == y_hat_train) / N
        assert acc_train == 1.0

        scores = classifier.predict_proba(W_test.T)

        yo.append(scores[:, 1])

        acc_test = np.sum(l_test.ravel() == scores) / K

    yo = np.array(yo)

    trues = np.argmax(yo, axis=0)

    y_hat = np.array(
        list(map(lambda x: list(classes)[x], trues))).reshape(1, -1)

    acc = np.sum(y_test == y_hat) / K

    print(acc * 100)

    a = y_test.ravel()
    b = y_hat.ravel()

    cnf_matrix = confusion_matrix(a, b, labels=list(classes))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')
    plt.savefig('data/out/cnf_matrix.pdf', format='pdf', dpi=300)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('data/out/cnf_matrix_norm.pdf', format='pdf', dpi=300)
