import numpy as np
from feedforward import BCPNN as FF_BCPNN

class BCPNN(FF_BCPNN):

    def __init__(self, max_iter=1e3, tol=1e-4, prob_threshold=0.5):
        self.MAX_ITER = max_iter
        self.TOL = tol
        self.PROB_THRESHOLD = prob_threshold

    def fit(self, X):
        n_samples, n_features = X.shape
        super().fit(X, X)

    def predict(self, X, return_binary=False):
        input = X
        prev = np.empty_like(X)
        iter = 0
        while  (iter < self.MAX_ITER and \
                not np.allclose(input, prev, atol=self.TOL)):
            prev = input
            input = self.predict_proba(prev)
            iter += 1
        if return_binary:
            input = self._proba_to_bin(input, self.PROB_THRESHOLD)
        return input

    def score(self, X, y):
        return (self.predict(X, return_binary=True) == y).sum() / X.size

    def _proba_to_bin(self, x, threshold):
        return np.where(x >= threshold, 1, 0)
