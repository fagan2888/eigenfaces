# scientific computing library
import numpy as np
# `.mat` to `Python`-compatible data converter
import scipy.io


def fetch_data(fname='face', ratio=0.8, seed=42):
    """Bootstrapping helper function for fetching data.

    Parameters
    ----------
    fname: str
        Name of the `.mat` input file
    ratio: float
        Split ratio of dataset
    seed: int
        Random seed initial state

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
    X = data['X']  # shape: [D x N]
    y = data['l']  # shape: [1 x N]

    assert(X.shape[1] == y.shape[1])
    # Number of images
    D, N = X.shape

    # Fix the random seed
    np.random.seed(seed)

    # Cardinality of labels
    _card = len(set(y.ravel()))

    # Step splitting of dataset
    _step = int(N / _card)

    # Shape boundaries
    _bounds = np.arange(0, N, _step)

    # Shapes
    shapes = list(zip(_bounds[:-1], _bounds[1:]))

    # Training Mask
    _mask = []

    for _shape in shapes:
        _idx = np.random.choice(
            np.arange(*_shape), int(ratio * _step), replace=False)
        _mask.append(_idx)

    mask_train = np.array(_mask).ravel()

    mask_test = np.array(list(set(np.arange(0, N)) - set(mask_train)))

    # Partition dataset to train and test sets
    X_train, X_test = X[:, mask_train], X[:, mask_test]
    y_train, y_test = y[:, mask_train], y[:, mask_test]

    return {'train': (X_train.T, y_train.T), 'test': (X_test.T, y_test.T)}
