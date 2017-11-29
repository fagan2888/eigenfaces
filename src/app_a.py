# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np

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

SHAPE = (46, 56)

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

    logger = logging.getLogger('app_a')

    if _level.upper() == 'INFO':
        coloredlogs.install(level='IFNO', logger=logger)
    elif _level.upper() == 'DEBUG':
        coloredlogs.install(level='DEBUG', logger=logger)
    else:
        coloredlogs.install(level='WARNING', logger=logger)

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

    # TODO
    # threshold w's
    logger.warning('TODO: threshold eigenvalues')

    # Sorted eigenvalues and eigenvectors
    l = _l[_indexes]
    logger.debug('l.shape=%s' % (l.shape,))
    v = _v[:, _indexes]
    logger.debug('v.shape=%s' % (v.shape,))

    M = np.arange(1, N + 1)

    error = []

    logger.info('Reconstruction for M in [%d,%d]...' % (M[0], M[-1]))
    for j, m in enumerate(M):

        progress(j + 1, len(M), status='Reconstruction for M=%d' % m)

        V = v[:, :m]

        _U = np.dot(A, V)

        U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)

        W = np.dot(U.T, A)

        A_hat = np.dot(U, W)

        error.append(np.mean(np.sum((A - A_hat)**2)))
    # TODO
    # fix bug of progress bar after '\r'
    print('')

    logger.info('Plotting reconstruction error versus M...')
    plt.plot(M, error)
    plt.savefig('data/out/error_versus_M.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/error_versus_M.pdf...')
