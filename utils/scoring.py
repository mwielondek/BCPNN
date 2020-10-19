from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer as KBD

from ..feedforward_modular import BCPNN as mBCPNN
from ..feedforward import BCPNN

import numpy as np

class Scorer:

    LIST_NB = [MultinomialNB(), BernoulliNB(), GaussianNB()]
    LIST_BCPNN = [BCPNN()]
    LIST_CLFS = LIST_NB + LIST_BCPNN

    def __init__(self, clfs=LIST_CLFS):
        self.clfs = clfs

    def score_all(self, X, y, **kwargs):
        scores = {}
        for clf in self.clfs:
            scores[str(clf)] = self.score(clf, X, y, **kwargs)
        return scores

    def pretty_print(self, scores):
        for k,v in scores.items():
            print("--- {:20} ---".format(k))
            print("Score: {:.3f} +/-{:.3f}".format(*v))

    def score(self, clf, X, y, folds=4, seed=0, preprocess='none', **kwargs):
        if preprocess == 'discretize':
            X = self.discretize(X, **kwargs)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=kf)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2

    def discretize(self, X, n_bins=5, strategy='uniform'):
        """Discretization (otherwise known as quantization or binning) provides a way
        to partition continuous features into discrete values."""

        # n_bins determines number of bins per feature
        discretizer = KBD(n_bins, encode='onehot-dense', strategy=strategy)

        Xb = discretizer.fit_transform(X)

        return Xb
