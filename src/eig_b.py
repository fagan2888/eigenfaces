# matplotlib backtest for missing $DISPLAY
import matplotlib
matplotlib.use('Agg')

# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [12.0, 9.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

# helper data preprocessor
from reader import fetch_data

# logging module
import logging
import coloredlogs

# argument parser
import argparse

# built-in tools
import time
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
        coloredlogs.install(level='IFNOR', logger=logger)
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
    plt.title('Mean Face\n')
    plt.savefig('data/out/mean_face_eig_b.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/mean_face_eig_b.pdf...')

    A = X_train - mean_face
    logger.debug('A.shape=%s' % (A.shape,))

    S = (1 / N) * np.dot(A.T, A)
    logger.debug('S.shape=%s' % (S.shape,))

    # Calculate eigenvalues `w` and eigenvectors `v`
    logger.info('Calculating eigenvalues and eigenvectors...')
    t = time.time()
    _w, _v = np.linalg.eig(S)
    _u = np.dot(A, _v)
    logger.error('Duration %.2fs' % (time.time() - t))

    # Indexes of eigenvalues, sorted by value
    logger.info('Sorting eigenvalues...')
    _indexes = np.argsort(np.abs(_w))[::-1]

    # TODO
    # threshold w's
    logger.warning('TODO: threshold eigenvalues')

    # Sorted eigenvalues and eigenvectors
    w = _w[_indexes]
    logger.debug('w.shape=%s' % (w.shape,))
    u = _u[:, _indexes]
    logger.debug('u.shape=%s' % (u.shape,))

    logger.info('Plotting eigenfaces...')
    n_images = 3
    fig, axes = plt.subplots(nrows=1, ncols=n_images)
    for ax, img, i in zip(axes.flatten(), u[:, :n_images].T, range(1, n_images + 1)):
        ax.imshow(img.reshape(SHAPE).T,
                  cmap=plt.cm.Greys)
        ax.set_title('Eigenface %d' % i)
    fig.savefig('data/out/eff_eigenfaces.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/eff_eigenfaces.pdf...')

    plt.figure(figsize=(8.0, 6.0))

    logger.info('Plotting sorted eigenvalues...')
    plt.bar(range(len(w)), np.abs(w))
    plt.title('Sorted Eigenvalues')
    plt.xlabel('$w_{m}$: $m^{th}$ eigenvalue')
    plt.ylabel('Real Value')
    plt.savefig('data/out/eigenvalues_eig_b.pdf',
                format='pdf', dpi=1000, transparent=True)
    logger.info('Exported at data/out/eigenvalues_eig_b.pdf...')
