import pandas as pd

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import ParameterGrid

from ..clusters import get_cluster_ids

class GridSearch:

    def fit(self, clf, X, y, params, fit_params={}, verbose=False, scoring_fn=ami, decimals=None):
        clf.fit(X, **fit_params)
        if verbose:
            print("Finished fitting")

        res = dict(params=[], score=[])
        params = ParameterGrid(params)
        for i, param_set in enumerate(params):
            if verbose:
                print("[{}/{}] Predicting with params {}".format(i+1, len(params), param_set))
            for k,v in param_set.items():
                setattr(clf, k, v)
            pred = clf.predict(X)
            clsid = get_cluster_ids(pred, decimals=decimals)
            score = scoring_fn(clsid, y)
            res['params'].append(param_set)
            res['score'].append(score)

        self.res_ = res

    def disp_res(self):
        return pd.DataFrame(self.res_).sort_values('score', ascending=False)
