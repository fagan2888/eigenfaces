# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [8.0, 6.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from reader import fetch_data
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
import psutil


if __name__ == '__main__':

    # argument parser instance
    parser = argparse.ArgumentParser()
    # init log level argument
    parser.add_argument('-l', '--log', type=str,
                        help='<optional> Log Level (info | debug)')
    parser.add_argument('-k', '--n_neighbors', type=int,
                        help='<optional> Number of nearest neighbors')
    # parse arguments
    argv = parser.parse_args()
    # get log level
    _level = argv.log or ''
    # get number of neighbors
    n_neighbors = argv.n_neighbors or 1

    logger = logging.getLogger(os.path.basename(__file__).replace('.py', ''))

    if _level.upper() == 'INFO':
        coloredlogs.install(level='IFNO', logger=logger)
    elif _level.upper() == 'DEBUG':
        coloredlogs.install(level='DEBUG', logger=logger)
    else:
        coloredlogs.install(level='WARNING', logger=logger)

    logger.debug('n_neighbors=%s' % n_neighbors)

    logger.info('Fetching data...')
    data = fetch_data()

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

    # Sorted eigenvalues and eigenvectors
    l = _l[_indexes]
    logger.debug('l.shape=%s' % (l.shape,))
    v = _v[:, _indexes]
    logger.debug('v.shape=%s' % (v.shape,))

    classes = set(y_train.ravel())

    Ms = np.arange(1, len(l))
    # value of M for confusion matrix
    M_star = Ms[-1]

    acc = []
    train_dur = []
    test_dur = []
    memory = []

    logger.info('Model Evaluation for M in [%d,%d]...' % (Ms[0], Ms[-1]))
    for j, M in enumerate(Ms):

        progress(j + 1, len(Ms), status='Model for M=%d' % M)

        # start timer
        _start = time.time()

        if _level.upper() == 'DEBUG':
            print('')
        logger.debug('M=%s' % M)

        V = v[:, :M]
        logger.debug('V.shape=%s' % (V.shape,))

        _U = np.dot(A, V)

        U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
        logger.debug('U.shape=%s' % (U.shape,))

        W = np.dot(U.T, A)
        logger.debug('W.shape=%s' % (W.shape,))

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(W.T, y_train.T.ravel())

        # stop timer
        _stop = time.time()

        # train time
        train_dur.append(_stop - _start)

        X_test, y_test = data['test']
        I, K = X_test.shape
        assert I == D, logger.error(
            'Number of features of test and train data do not match!')
        logger.debug('Number of features: D=%d' % I)
        logger.debug('Number of test data: K=%d' % K)

        accuracy = 0

        # start timer
        _start = time.time()

        Phi = X_test - mean_face
        logger.debug('Phi.shape=%s' % (Phi.shape,))

        W_test = np.dot(U.T, Phi)
        logger.debug('W_test.shape=%s' % (W_test.shape,))

        W_test = np.dot(Phi.T, U)

        y_hat = classifier.predict(W_test)

        accuracy = 100 * np.sum(y_test.ravel() == y_hat) / K

        # stop timer
        _stop = time.time()

        # store values for confusion matrix
        if M == M_star:
            cnf_matrix = confusion_matrix(
                y_test.ravel(), y_hat, labels=list(classes))

        # test time
        test_dur.append(_stop - _start)

        # pct memory usage
        memory.append(psutil.Process(os.getpid()).memory_percent())

        # TODO
        # fix bug of progress bar after '\r'
        acc.append(accuracy)

    logger.error('Best Accuracy = %.2f%%' % (np.max(acc)))

    if _level.upper() == 'INFO':
        print('')
    logger.info('Plotting recognition accuracy versus M...')
    plt.plot(Ms, acc)
    plt.title(
        'Recognition Accuracy versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Recognition Accuracy [%]')
    plt.savefig('data/out/accuracy_versus_M.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/accuracy_versus_M.pdf...')

    logger.info('Plotting time versus M...')
    plt.figure()
    # plt.plot(Ms, train_dur)
    sns.regplot(x=Ms.reshape(-1, 1), y=np.array(train_dur))
    # plt.plot(Ms, test_dur)
    sns.regplot(x=Ms.reshape(-1, 1), y=np.array(test_dur))
    plt.title('Execution Time versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Execution Time [sec]')
    plt.legend(['Train', 'Test'])
    plt.savefig('data/out/time_versus_M.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info(
        'Exported at data/out/time_versus_M.pdf...')

    logger.info('Plotting memory versus M...')
    plt.figure()
    plt.plot(Ms, memory)
    #sns.regplot(x=Ms.reshape(-1, 1), y=np.array(memory))
    plt.title('Memory Percentage Usage versus $\mathcal{M}$\n')
    plt.xlabel('$\mathcal{M}$: number of principle components')
    plt.ylabel('Memory Usage [%]')
    plt.savefig('data/out/memory_versus_M.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info(
        'Exported at data/out/memory_versus_M.pdf...')

    plt.rcParams['figure.figsize'] = [28.0, 21.0]

    # Plot non-normalized confusion matrix
    plt.figure()
    logger.info('Plotting confusion matrices...')
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Nearest Neighbor - Confusion Matrix',
                          cmap=plt.cm.Greens)
    plt.savefig('data/out/nn_cnf_matrix.pdf', format='pdf', dpi=300)
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Nearest Neighbor - Normalized Confusion Matrix',
                          cmap=plt.cm.Greens)
    plt.savefig('data/out/nn_cnf_matrix_norm.pdf', format='pdf', dpi=300)
    logger.info(
        'Exported at data/out/nn_cnf_matrix.pdf & data/out/nn_cnf_matrix_norm.pdf...')
