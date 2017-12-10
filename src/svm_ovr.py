# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np
from sklearn.svm import SVC
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

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# built-in tools
import time
import os
from functools import reduce
np.warnings.filterwarnings('ignore')


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
    parser.add_argument('-cv', '--cross_validation', action='store_true',
                        help='<optional> Cross validate SVM')
    # parse arguments
    argv = parser.parse_args()
    # get log level
    _level = argv.log or ''
    # get number of principle components
    M = argv.n_comps or 121
    # get flag of standardization
    standard = argv.standard or False
    # get flag of cross validation
    cv = argv.cross_validation or False

    logger = logging.getLogger(os.path.basename(__file__).replace('.py', ''))

    if _level.upper() == 'INFO':
        coloredlogs.install(level='IFNO', logger=logger)
    elif _level.upper() == 'DEBUG':
        coloredlogs.install(level='DEBUG', logger=logger)
    else:
        coloredlogs.install(level='WARNING', logger=logger)

    logger.debug('standard=%s' % standard)
    logger.debug('cross_validation=%s' % cv)

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

    if cv:
        # cross validation grid
        params = {
            'gamma': np.logspace(-5, 3, 5), 'kernel': ['rbf', 'linear'], 'C': np.logspace(0, 2, 3)}
        mean_fit_time = {k: 0 for k in params['kernel']}
        mean_score_time = {k: 0 for k in params['kernel']}
        mean_n_support_ = 0

    for c in classes:

        # preprocess training labels
        l_train = -np.ones(y_train.T.shape)
        l_train[y_train.T == c] = 1

        _classifier = SVC(kernel='linear', C=1, probability=True, gamma=2e-4)

        if cv:
            search = GridSearchCV(_classifier, params, n_jobs=-1)

            search.fit(W_train.T, l_train.ravel())

            classifier = search.best_estimator_

            _results = list(zip(search.cv_results_['params'],
                                search.cv_results_['mean_fit_time'],
                                search.cv_results_['mean_score_time']))

            for kernel in params['kernel']:
                _f = filter(lambda x: kernel == x[0]['kernel'], _results)
                for _, fit_time, score_time in _f:
                    mean_fit_time[kernel] += fit_time
                    mean_score_time[kernel] += score_time
                mean_fit_time[kernel] /= len(search.cv_results_['params']
                                             ) / len(params['kernel'])
                mean_score_time[kernel] /= len(
                    search.cv_results_['params']) / len(params['kernel'])

            mean_n_support_ += np.sum(classifier.n_support_)

        else:
            classifier = _classifier
            classifier.fit(W_train.T, l_train.ravel())

        probs = classifier.predict_proba(W_test.T)

        scores.append(probs[:, 1])

    scores = np.array(scores)

    if cv:
        mean_n_support_ /= len(classes)
        logger.error('Mean `fit`   Time %s' % mean_fit_time)
        logger.error('Mean `score` Time %s' % mean_score_time)
        logger.error('Mean Number of Support Vectors %s' % mean_n_support_)

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
