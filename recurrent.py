import numpy as np
from feedforward import BCPNN as FF_BCPNN

class BCPNN(FF_BCPNN):

    def __init__(self, predict_mode='rec', max_iter=1e3, tol=1e-4):
        # Recursive version shows slight improvement using timeit
        if predict_mode == 'rec':
            self.predict = self._predict_rec
        elif predict_mode == 'iter':
            self.predict = self._predict_iter

        self.MAX_ITER = max_iter
        self.TOL = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        super().fit(X, X)
        print(self.n_classes_)

    def _predict_iter(self, X):
        input = X
        prev = np.empty_like(X)
        iter = 0
        while iter < self.MAX_ITER and not np.allclose(input, prev, atol=self.TOL):
            prev = input
            input = self.predict_proba(prev)
            iter += 1
        return input

    def _predict_rec(self, X, iter=0):
        if iter >= self.MAX_ITER:
            return X
        next = self.predict_proba(X)
        if np.allclose(X, next, atol=self.TOL):
            return next
        return self._predict_rec(next, iter + 1)
