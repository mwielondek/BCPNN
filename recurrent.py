import numpy as np
from feedforward import BCPNN as ffBCPNN

class rBCPNN(ffBCPNN):
    """ A recurrent version of the Bayesian Confidence
    Propagation Neural Network (BCPNN).

    @author M. Wielondek
    """

    def __init__(self, max_iter=1e3, tol=1e-4, prob_threshold=0.5, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.MAX_ITER = max_iter
        self.TOL = tol
        self.PROB_THRESHOLD = prob_threshold
        self.VERBOSE = verbose

    def fit(self, X):
        n_samples, n_features = X.shape
        super().fit(X, X)

    def predict(self, X, return_binary=False):
        input = X
        prev = np.zeros_like(X)
        iter = 0
        while  (iter < self.MAX_ITER and \
                not np.allclose(input, prev, atol=self.TOL, rtol=0)):
            prev = input
            input = self.predict_proba(prev)
            iter += 1
        if return_binary:
            input = self._binarize(input, self.PROB_THRESHOLD)
        if self.VERBOSE and iter >= self.MAX_ITER:
            print("BCPNN: reached max iteration limit of ", self.MAX_ITER)
        return input

    def score(self, X, y, tol=0.5):
        """Returns the accuracy score for the classifier. Return value cosists
        of two averages; first counts correct number of separate features,
        second the number of fully recovered samples."""
        close = np.isclose(self.predict(X), y, atol=tol, rtol=0)
        return {'features': close.sum() / X.size,
                'samples': np.all(close, axis = 1).sum() / X.shape[0]}

    def _binarize(self, x, threshold=0.5):
        return np.where(x >= threshold, 1, 0)
