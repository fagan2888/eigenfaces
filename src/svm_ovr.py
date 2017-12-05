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

# built-in tools
import time
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

    pca = PCA(n_comps=M, standard=standard, logger=logger)
    logger.info('Applying PCA with M=%d' % M)

    W_train = pca.fit(X_train)
    logger.debug('W_train.shape=%s' % (W_train.shape,))

    X_test, y_test = data['test']
    I, K = X_test.shape
    assert I == D, logger.error(
        'Number of features of test and train data do not match, %d != %d' % (D, I))

    W_test = pca.transform(X_test)
    logger.debug('W_test.shape=%s' % (W_test.shape,))

    classes = set(y_train.ravel())

    scores = []

    for c in classes:

        # preprocess training labels
        l_train = -np.ones(y_train.T.shape)
        l_train[y_train.T == c] = 1
        # preprocess testing labels
        l_test = -np.ones(y_test.T.shape)
        l_test[y_test.T == c] = 1

        # search.best_estimator_

        classifier = SVC(kernel='linear', C=10, probability=True, gamma=2e-4)

        classifier.fit(W_train.T, l_train.ravel())

        if classifier.kernel != 'rbf':
            y_hat_train = classifier.predict(W_train.T)
            acc_train = np.sum(l_train.ravel() == y_hat_train) / N
            # make sure 100% training accuracy
            assert acc_train == 1.0

        probs = classifier.predict_proba(W_test.T)

        scores.append(probs[:, 1])

        acc_test = np.sum(l_test.ravel() == probs) / K

    scores = np.array(scores)

    trues = np.argmax(scores, axis=0)

    y_hat = np.array(
        list(map(lambda x: list(classes)[x], trues))).reshape(1, -1)

    acc = np.sum(y_test == y_hat) / K

    logger.error('Accuracy = %.2f%%' % (acc * 100))

    cnf_matrix = confusion_matrix(
        y_test.ravel(), y_hat.ravel(), labels=list(classes))

    # Plot non-normalized confusion matrix
    plt.figure()
    logger.info('Plotting confusion matrices...')
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='SVM One versus Rest - Confusion Matrix',
                          cmap=plt.cm.Blues)
    plt.savefig('data/out/svm_ovr_cnf_matrix.pdf', format='pdf', dpi=300)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='SVM One versus Rest - Normalized Confusion Matrix',
                          cmap=plt.cm.Blues)
    plt.savefig('data/out/svm_ovr_cnf_matrix_norm.pdf', format='pdf', dpi=300)
    logger.info(
        'Exported at data/out/svm_ovr_cnf_matrix.pdf & data/out/svm_ovr_cnf_matrix_norm.pdf...')
