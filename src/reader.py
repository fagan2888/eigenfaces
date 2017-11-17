import numpy as np
import scipy.io


def fetch_data(fname='face.mat', ratio=0.8):
    """Bootstrapping helper function for fetching data.

    Parameters
    ----------
    fname: str
        Name of the `.mat` input file
    ratio: float
        Split ratio of dataset

    Returns
    -------
    data: dict
        * train: tuple
            - X: features
            - y: labels
        * test: tuple
            - X: features
            - y: labels
    """

    # load `.mat` file
    data = scipy.io.loadmat('data/%s' % fname)

    # Images
    # N: number of images
    # K: number of pixels
    # shape: [N x K]
    _X = data['X'].T
    _y = data['l'].T

    assert(_X.shape[0] == _y.shape[0])
    # Number of images
    N = _X.shape[0]

    # Fix the random seed
    np.random.seed(0)

    # Shuffled indeces
    _mask = np.arange(0, N)
    np.random.shuffle(_mask)

    # Randomised data
    X = _X[_mask]
    y = _y[_mask]

    # Ratition dataset to train and test sets
    X_train, X_test = X[:int(N * ratio)], X[int(N * ratio):]
    y_train, y_test = y[:int(N * ratio)], y[int(N * ratio):]

    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
