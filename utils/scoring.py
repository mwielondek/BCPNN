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
        pipeline_kwargs = {k.split('__', 1)[1]: v for k,v in kwargs.items() if k.split('__', 1)[0] == 'pipeline'}
        cv_score_kwargs = {k.split('__', 1)[1]: v for k,v in kwargs.items() if k.split('__', 1)[0] == 'cv_score'}

        pipe = self.create_pipeline(clf, **pipeline_kwargs)

        # if OneHotEncoder is one of the steps we need to prerun the pipe due to how
        # cross_val_score clones the estimators making us lose module_sizes_ attribute
        if 'onehot-encode' in pipeline_kwargs.get('preprocess', ''):
            pipe[:-1].fit(X,y)

        if isinstance(clf, mBCPNN):
            print("mbcpnn!")
            modsz = lambda: pipe.named_steps['onehot-encoder'].module_sizes_
            cv_score_kwargs.update(fit_params=dict(clf__module_sizes=modsz))

        return self.cv_score(pipe, X, y, **cv_score_kwargs)

    def cv_score(self, pipe, X, y, folds=4, seed=0, fit_params={}):
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=kf, fit_params=fit_params)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2
