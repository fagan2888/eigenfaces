# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [8.0, 6.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")
sns_b, sns_g, sns_r, sns_v, sns_y, sns_l = sns.color_palette("muted")


# helper data preprocessor
from reader import fetch_data

# utility functions
from utils import progress

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# built-in tools
import os

SHAPE = (46, 56)

if __name__ == '__main__':

    # argument parser instance
    parser = argparse.ArgumentParser()
    # init log level argument
    parser.add_argument('-l', '--log', type=str,
                        help='<optional> Log Level (info | debug)')
    # parse arguments
    argv = parser.parse_args()
    # get log level
    _level = argv.log or ''

    logger = logging.getLogger(os.path.basename(__file__).replace('.py', ''))

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
    # fix bug of progress bar after '\r'
    print('')

    logger.info('Plotting reconstruction error versus M...')
    fig, ax1 = plt.subplots()

    lns1 = ax1.plot(M, error, color=sns_b, label='Reconstruction Error')
    ax1.tick_params('y', colors=sns_b)

    ax2 = ax1.twinx()
    lns2 = ax2.plot(M, l, color=sns_g, label='Covariance Matrix Eigenvalues')
    ax2.tick_params('y', colors=sns_g)

    ax1.set_title(
        'Reconstruction Error versus Number of Principle Components $\mathcal{M}$\n')
    ax1.set_xlabel('$\mathcal{M}$: Number of Principle Components')
    ax1.set_ylabel('$\mathcal{J}$: Reconstruction Error')
    ax2.set_ylabel('Covariance Matrix Eigenvalues')
    # fix legend hack
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    # ax1.grid()
    fig.tight_layout()
    plt.savefig('data/out/error_versus_M.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/error_versus_M.pdf...')

    # set M
    m = 100
    V = v[:, :m]
    _U = np.dot(A, V)
    U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)
    W_train = np.dot(U.T, A)

    # test data
    X_test, y_test = data['test']
    I, K = X_test.shape
    assert I == D, logger.error(
        'Number of features of test and train data do not match, %d != %d' % (D, I))
    Phi = X_test - mean_face
    logger.debug('Phi.shape=%s' % (Phi.shape,))

    W_test = np.dot(U.T, Phi)
    logger.debug('W_test.shape=%s' % (W_test.shape,))

    ridx_train = np.random.randint(0, N, 3)
    R_train = W_train[:, ridx_train]
    B_train = np.dot(U, R_train)

    plt.rcParams['figure.figsize'] = [16.0, 12.0]

    logger.info('Plotting reconstructed training images...')
    fig, axes = plt.subplots(nrows=2, ncols=3)
    titles_train = ['Original Train', 'Original Train', 'Original Train',
                    'Reconstructed Train', 'Reconstructed Train', 'Reconstructed Train']
    for ax, img, title in zip(axes.flatten(), np.concatenate((A[:, ridx_train], B_train), axis=1).T, titles_train):
        _img = img + mean_face.ravel()
        ax.imshow(_img.reshape(SHAPE).T,
                  cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax.set_title(title)
    fig.savefig('data/out/reconstructed_train_images.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/reconstructed_train_images.pdf...')

    ridx_test = np.random.randint(0, K, 3)
    R_test = W_test[:, ridx_test]
    B_test = np.dot(U, R_test)

    logger.info('Plotting reconstructed testing images...')
    fig, axes = plt.subplots(nrows=2, ncols=3)
    titles_test = ['Original Test', 'Original Test', 'Original Test',
                   'Reconstructed Test', 'Reconstructed Test', 'Reconstructed Test']
    for ax, img, title in zip(axes.flatten(), np.concatenate((Phi[:, ridx_test], B_test), axis=1).T, titles_test):
        _img = img + mean_face.ravel()
        ax.imshow(_img.reshape(SHAPE).T,
                  cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax.set_title(title)
    fig.savefig('data/out/reconstructed_test_images.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/reconstructed_test_images.pdf...')
