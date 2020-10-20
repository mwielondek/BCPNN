from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer as KBD, StandardScaler

from ..feedforward_modular import BCPNN as mBCPNN
from ..feedforward import BCPNN


from sklearn.pipeline import Pipeline
from ..encoder import ComplementEncoder, OneHotEncoder

class Scorer:

    LIST_NB = [MultinomialNB(), BernoulliNB(), GaussianNB()]
    LIST_BCPNN = [BCPNN()]
    LIST_CLFS = LIST_NB + LIST_BCPNN

    def __init__(self, clfs=LIST_CLFS):
        self.clfs = clfs

    def score_all(self, X, y, **kwargs):
        scores = {}
        for clf in self.clfs:
            scores[str(clf)] = self.score_helper(clf, X, y, **kwargs)
        return scores

    def pretty_print(self, scores):
        for k,v in scores.items():
            print("--- {:20} ---".format(k))
            print("Score: {:.3f} +/-{:.3f}".format(*v))

    def create_pipeline(self, clf, preprocess=(), pipeline_params={}):
        estimators = []
        if 'scale' in preprocess:
            estimators.append(('scaler', StandardScaler(with_std=True, with_mean=False)))

        if 'discretize' in preprocess:
            estimators.append(('discretizer', KBD(n_bins=5, encode='ordinal', strategy='uniform')))

        if 'onehot-encode' in preprocess:
            estimators.append(('onehot-encoder', OneHotEncoder()))

        if 'complement-encode' in preprocess:
            estimators.append(('complement-encoder', ComplementEncoder()))

        estimators.append(('clf', clf))

        pipe = Pipeline(estimators)
        pipe.set_params(**pipeline_params)

        return pipe

    def score_helper(self, clf, X, y, **kwargs):
        pipe = self.create_pipeline(clf, **kwargs)
        return self.cv_score(pipe, X, y)

    def cv_score(self, pipe, X, y, folds=4, seed=0, fp={}):
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=kf, fit_params=fp)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2
