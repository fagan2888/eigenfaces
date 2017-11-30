# scientific computing library
import numpy as np
# `.mat` to `Python`-compatible data converter
import scipy.io

from sklearn.model_selection import train_test_split


def fetch_data(fname='face', ratio=0.8):
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
    data = scipy.io.loadmat('data/%s.mat' % fname)

    # Images
    # N: number of images
    # D: number of pixels
    _X = data['X']  # shape: [D x N]
    _y = data['l']  # shape: [1 x N]

    assert(_X.shape[1] == _y.shape[1])
    # Number of images
    D, N = _X.shape

    X_train, X_test, y_train, y_test = train_test_split(
        _X.T, _y.T, test_size=0.2, random_state=11)

    # # Fix the random seed
    # np.random.seed(11)

    # # Shuffled indeces
    # _mask = np.arange(N)
    # np.random.shuffle(_mask)

    # # Randomised data
    # X = _X[:, _mask]
    # y = _y[:, _mask]

    # # Ratition dataset to train and test sets
    # X_train, X_test = X[:, :int(N * ratio)], X[:, int(N * ratio):]
    # y_train, y_test = y[:, :int(N * ratio)], y[:, int(N * ratio):]

    return {'train': (X_train.T, y_train.T), 'test': (X_test.T, y_test.T)}
