from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer as KBD

import warnings

from BCPNN.encoder import BinvecOneHotEncoder as Encoder

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


## OLD BELOW


def score_clf_m(clf, X, y, n_bins=4, folds=4, seed=None, encode=False):
    def fit_score(X,y,Xtest=None,ytest=None):
        if Xtest is None:
            Xtest = X
            ytest = y
        # clf.fit(X,y)
        # return clf.score(Xtest,ytest)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        m_sz = np.hstack((np.full(X.shape[1], 2), np.unique(y).size))
        fp = {'module_sizes': m_sz} #if type(clf) == mBCPNN else None
        scores = cross_val_score(clf, Xtest, ytest, cv=kf, fit_params=fp)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2

    print("---- {} ----".format(clf))

    fstr = "{:.3f} +/-{:.3f}"

    Xp, _ = prepdata(X, n_bins)
    Xe = Encoder.transform(Xp)
    s = fit_score(Xe,y)
    print("Score on fit/score(X_binned_encoded, y):\t\t\t "+fstr.format(*s))
    print()

    return s

def score_clf(clf, X, y, n_bins=4, folds=4, seed=None, encode=False):
    modular_clf = type(clf) == mBCPNN

    def fit_score(X,y):
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        m_sz = np.hstack((np.full(X.shape[1] // 2, 2), np.unique(y).size))
        fp = {'module_sizes': m_sz} if type(clf) == mBCPNN else None
        scores = cross_val_score(clf, X, y, cv=kf, fit_params=fp)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2

    print("---- {} ----".format(clf))

    fstr = "{:.3f} +/-{:.3f}"

    if not modular_clf:
        s = fit_score(X,y)
        print("Score on fit/score(X, y):\t\t\t\t\t "+fstr.format(*s))

        Xp, _ = prepdata(X, n_bins)
        s = fit_score(Xp,y)
        print("Score on fit/score(X_binned, y):\t\t\t\t "+fstr.format(*s))

        if encode:
            Xe = Encoder.transform(Xp)
            s = fit_score(Xe,y)
            print("Score on fit/score(X_binned_encoded, y):\t\t\t "+fstr.format(*s))
    else:
        Xp, _ = prepdata(X, n_bins)
        Xe = Encoder.transform(Xp)
        s = fit_score(Xe,y)
        print("Score on fit/score(X_binned_encoded, y):\t\t\t "+fstr.format(*s))

    print()

    return s

def score_all(**kwargs):
    for clf in [BCPNN(), mBCPNN(), MultinomialNB(), BernoulliNB(), GaussianNB()]:
        score_clf(clf, X, y, **kwargs)

def collect_results(i=5, **kwargs):
    results = {}
    for clf in [BCPNN(), mBCPNN(), MultinomialNB(), BernoulliNB(), GaussianNB()]:
        s = [score_clf(clf, X, y, seed=n, **kwargs) for n in range(i)]
        results[str(clf)[:24]] = s
    return results

def print_results(results):
    for k,v in results.items():
        print("--- {} ---".format(k))
        r=np.array(v).mean(axis=0)
        print("Score: {:.3f} +/-{:.3f}".format(*r))
