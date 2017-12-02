# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

#timing utility
import time

# prettify plots
plt.rcParams['figure.figsize'] = [20.0, 15.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from reader import fetch_data

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

    logger.info('Plotting mean face...')
    plt.imshow(mean_face.reshape(SHAPE).T,
               cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    plt.savefig('data/out/mean_face_eig_a.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/mean_face_eig_a.pdf...')

    A = X_train - mean_face
    logger.debug('A.shape=%s' % (A.shape,))

    S = (1 / N) * np.dot(A, A.T)
    logger.debug('S.shape=%s' % (S.shape,))

    # Calculate eigenvalues `w` and eigenvectors `v`
    logger.info('Calculating eigenvalues and eigenvectors...')

    t = time.process_time()
    _w, _v = np.linalg.eig(S)
    print(time.process_time() - t)

    # Indexes of eigenvalues, sorted by value
    logger.info('Sorting eigenvalues...')
    _indexes = np.argsort(np.abs(_w))

    # TODO
    # threshold w's
    logger.warning('TODO: threshold eigenvalues')

    # Sorted eigenvalues and eigenvectors
    w = _w[_indexes]
    logger.debug('w.shape=%s' % (w.shape,))
    v = _v[:, _indexes]
    logger.debug('v.shape=%s' % (v.shape,))

    plt.figure()

    logger.info('Plotting sorted eigenvalues...')
    plt.bar(range(len(w)), np.abs(w[::-1]))
    plt.savefig('data/out/eigenvalues_eig_a.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/eigenvalues_eig_a.pdf...')
