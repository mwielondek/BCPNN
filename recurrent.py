import numpy as np
from feedforward import BCPNN as FF_BCPNN

class BCPNN(FF_BCPNN):

    def __init__(self, predict_mode='rec'):
        # Recursive version shows slight improvement using timeit
        if predict_mode == 'rec':
            self.predict = self._predict_rec
        elif predict_mode == 'iter':
            self.predict = self._predict_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        super().fit(X, X)
        print(self.n_classes_)

    def _predict_iter(self, X):
        input = X
        prev = np.empty_like(X)
        while not np.allclose(input, prev):
            prev = input
            input = self.predict_proba(prev)
        return input

    def _predict_rec(self, X):
        next = self.predict_proba(X)
        if np.allclose(X, next):
            return next
        return self._predict_rec(next)
