# scientific computing library
import numpy as np


class PCA(object):
    """Principle Component Analysis."""

    def __init__(self, n_comps=5, logger=None):
        """Contructor.

        Parameters
        ----------
        n_comps: int
            Number of principle components
        """
        self._fitted = False
        self.n_comps = n_comps
        self.logger = logger
        self.mean = None
        self.U = None

    def fit(self, X):
        """Fit PCA according to `X.cov()`.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix

        Returns
        -------
        array: numpy.ndarray
            Transformed features matrix
        """
        if self._fitted:
            raise AssertionError('Already fitted.')
        self.D, N = X.shape
        if self.logger:
            self.logger.debug('Number of features: D=%d' % self.D)
            self.logger.debug('Number of train data: N=%d' % N)
        self.mean = X.mean(axis=1).reshape(-1, 1)
        # center data
        A = X - self.mean
        if self.logger:
            self.logger.debug('A.shape=%s' % (A.shape,))
        # covariance matrix
        S = (1 / N) * np.dot(A.T, A)
        if self.logger:
            self.logger.debug('S.shape=%s' % (S.shape,))

        if self.logger:
            self.logger.info('Calculating eigenvalues and eigenvectors...')
        _l, _v = np.linalg.eig(S)

        if self.logger:
            self.logger.info('Sorting eigenvalues...')
        _indexes = np.argsort(_l)[::-1]

        # Sorted eigenvalues and eigenvectors
        l, v = _l[_indexes], _v[:, _indexes]
        if self.logger:
            self.logger.debug('l.shape=%s' % (l.shape,))
        if self.logger:
            self.logger.debug('v.shape=%s' % (v.shape,))

        V = v[:, :self.n_comps]

        _U = np.dot(A, V)

        self.U = _U / np.apply_along_axis(np.linalg.norm, 0, _U)

        W = np.dot(self.U.T, A)

        self._fitted = True

        return W

    def transform(self, X):
        """Transform `X` by projecting it to PCA feature space.

        Parameters
        ----------
        X: numpy.ndarray
            Features matrix

        Returns
        -------
        array: numpy.ndarray
            Transformed features matrix
        """
        if not self._fitted:
            raise AssertionError('Not fitted yet.')

        Phi = X - self.mean
        if self.logger:
            self.logger.debug('Phi.shape=%s' % (Phi.shape,))

        W = np.dot(self.U.T, Phi)
        if self.logger:
            self.logger.debug('W.shape=%s' % (W.shape,))

        return W
