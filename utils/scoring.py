from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer as KBD, StandardScaler

from ..feedforward_modular import BCPNN as mBCPNN
from ..feedforward import BCPNN


from sklearn.pipeline import Pipeline

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

    def score(self, clf, X, y, folds=4, seed=0, preprocess=(), **kwargs):
        estimators = []
        if 'scale' in preprocess:
            estimators.append(('scaler', StandardScaler(with_std=True, with_mean=False)))
        if 'discretize' in preprocess:
            estimators.append(('discretizer', KBD(5, encode='onehot-dense', strategy='uniform')))
        estimators.append(('clf', clf))
        pipe = Pipeline(estimators)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=kf)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2
