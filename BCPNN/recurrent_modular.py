import numpy as np
from .feedforward_modular import mBCPNN

class rmBCPNN(mBCPNN):
    """ A recurrent version of the Bayesian Confidence
    Propagation Neural Network (BCPNN).

    @author M. Wielondek
    """

    def __repr__(self):
        return "rmBCPNN()"

    def __init__(self, max_iter=1e3, tol=1e-4, prob_threshold=0.5, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.MAX_ITER = max_iter
        self.TOL = tol
        self.PROB_THRESHOLD = prob_threshold
        self.VERBOSE = verbose

    def fit(self, X, module_sizes=None):
        n_samples, n_features = X.shape

        # correct module_sizes unless set manually; the default implementation sets one big output module
        if module_sizes is None:
            # n_features//2 for X and again for Y
            module_sizes = np.full(n_features, 2)
        super().fit(X, X, module_sizes=module_sizes)

    def predict(self, X, return_binary=False):
        input = X
        prev = np.zeros_like(X)
        iter = 0
        while  (iter < self.MAX_ITER and \
                not np.allclose(input, prev, atol=self.TOL, rtol=0)):
            prev = input
            input = self.predict_proba(prev, assert_off=True)
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

    def get_params(self, deep=True):
        ff_params = super().get_params(deep)
        r_params = {"prob_threshold": self.PROB_THRESHOLD,
                    "verbose": self.VERBOSE,
                    "tol": self.TOL,
                    "max_iter": self.MAX_ITER}
        return dict(**ff_params, **r_params)
