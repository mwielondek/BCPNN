from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer as KBD, MinMaxScaler

from ..feedforward_modular import BCPNN as mBCPNN
from ..feedforward import BCPNN


from sklearn.pipeline import Pipeline
from ..encoder import ComplementEncoder, OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)

class Scorer:

    def __init__(self, clfs_indices=None):
        LIST_CLFS = self._get_clf_list()
        if clfs_indices is not None:
            select_clfs = list(map(LIST_CLFS.__getitem__, clfs_indices))
            self.clfs = select_clfs
        else:
            self.clfs = LIST_CLFS

    def _get_clf_list(self):
        patched_mbcpnn = mBCPNN()

        def additional_setup(**kwargs):
            X, y, pipe, cv_score_kwargs = list(map(kwargs.get, ['X', 'y', 'pipe', 'cv_score_kwargs']))

            modsz = None
            # if OneHotEncoder is one of the steps we need to prerun the pipe due to how
            # cross_val_score clones the estimators making us lose module_sizes_ attribute
            if 'onehot_encoder' in pipe.named_steps.keys():
                pipe[:-1].fit(X,y)
                modsz = lambda: pipe.named_steps['onehot_encoder'].module_sizes_

            cv_score_kwargs.update(fit_params=dict(clf__module_sizes=modsz))

        patched_mbcpnn.additional_setup = additional_setup

        LIST_NB = [MultinomialNB(), BernoulliNB(), GaussianNB()]
        LIST_BCPNN = [BCPNN(), patched_mbcpnn]
        LIST_CLFS = LIST_NB + LIST_BCPNN

        return LIST_CLFS

    def score_all(self, X, y, method='pipeline', **kwargs):
        scores = {}
        for clf in self.clfs:
            if method == 'pipeline':
                f = self.score_pipeline
                scores[str(clf)] = f(clf, X, y, **kwargs)
            elif method == 'cv':
                f = self.cv_score
                # extract clf-specific fit_params
                kwfp = kwargs['fit_params']
                kwfp_copy = kwfp.copy()
                clf_fit_params = {k.split('__', 1)[1]: v for k,v in kwfp.items() if k.split('__', 1)[0] == str(clf)[:-2]}
                for k in list(kwfp.keys()):
                    if '__' in k:
                        del kwfp[k]
                kwfp.update(clf_fit_params)
                scores[str(clf)] = f(clf, X, y, **kwargs)
                kwargs['fit_params'] = kwfp_copy
        return scores

    def pretty_print(self, scores, mode='horizontal', print_best=False):
        outputstr = ""
        clfline = ""
        scoreline = ""
        tabs = "\t"*1
        for k,v in scores.items():
            clfstr = "--- {:17}".format(k)
            scorestr = "Score: {:.3f} +/-{:.3f}".format(*v)
            if mode == 'vertical':
                outputstr += "{}\n{}\n".format(clfstr, scorestr)
            elif mode == 'horizontal':
                clfline += "{}{}".format(clfstr, tabs)
                scoreline += "{}{}".format(scorestr, tabs)

        if mode == 'vertical':
            print(outputstr)
        elif mode == 'horizontal':
            print("{}\n{}".format(clfline, scoreline))
        if print_best:
            print("\n--> Best:", max(scores.items(), key=lambda x: x[1][0])[0])

    def create_pipeline(self, clf, preprocess=(), pipeline_params={}):
        estimators = []
        if 'scale' in preprocess:
            estimators.append(('scaler', MinMaxScaler()))

        if 'discretize' in preprocess:
            estimators.append(('discretizer', KBD(n_bins=5, encode='ordinal', strategy='uniform')))

        if 'onehot_encode' in preprocess:
            estimators.append(('onehot_encoder', OneHotEncoder()))

        if 'complement_encode' in preprocess:
            estimators.append(('complement_encoder', ComplementEncoder()))

        estimators.append(('clf', clf))

        pipe = Pipeline(estimators)
        pipe.set_params(**pipeline_params)

        return pipe

    def score_pipeline(self, clf, X, y, **kwargs):
        pipeline_kwargs = {k.split('__', 1)[1]: v for k,v in kwargs.items() if k.split('__', 1)[0] == 'pipeline'}
        cv_score_kwargs = {k.split('__', 1)[1]: v for k,v in kwargs.items() if k.split('__', 1)[0] == 'cv_score'}

        pipe = self.create_pipeline(clf, **pipeline_kwargs)

        # For special clf that need additional setup or fit params, like mBCPNN
        try:
            clf.additional_setup(X=X, y=y, pipe=pipe, cv_score_kwargs=cv_score_kwargs)
        except AttributeError:
            pass

        return self.cv_score(pipe, X, y, **cv_score_kwargs)

    def cv_score(self, clf, X, y, folds=4, seed=0, fit_params={}):
        kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=kf, fit_params=fit_params)
        # std times two for a 95% confidence level
        return scores.mean(), scores.std() * 2
