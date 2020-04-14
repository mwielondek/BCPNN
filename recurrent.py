import numpy as np
from feedforward import BCPNN as FF_BCPNN

class BCPNN(FF_BCPNN):

    def __init__(self, predict_mode='rec', max_iter=1e3, tol=1e-4):
        self.MAX_ITER = max_iter
        self.TOL = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        super().fit(X, X)

    def predict(self, X):
        input = X
        prev = np.empty_like(X)
        iter = 0
        while  (iter < self.MAX_ITER and \
                not np.allclose(input, prev, atol=self.TOL)):
            prev = input
            input = self.predict_proba(prev)
            iter += 1
        return input
