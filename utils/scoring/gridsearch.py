import pandas as pd

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import ParameterGrid

from ..clusters import get_cluster_ids

class GridSearch:

    decimals_key = '__decimals'
    mode_key = '__mode'

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
                if k in [self.decimals_key, self.mode_key]:
                    continue
                setattr(clf, k, v)
            pred = clf.predict(X)

            if self.decimals_key in param_set.keys():
                decimals = param_set[self.decimals_key]

            mode_param = {}
            if self.mode_key in param_set.keys():
                mode_param = dict(mode=param_set[self.mode_key])

            clsid = get_cluster_ids(pred, decimals=decimals, **mode_param)
            score = scoring_fn(clsid, y)
            res['params'].append(param_set)
            res['score'].append(score)

        self.res_ = res

    def get_res(self):
        return pd.DataFrame(self.res_).sort_values('score', ascending=False)

    def disp_res(self):
        with pd.option_context('display.max_colwidth', None, 'display.float_format', '{:.2%}'.format):
            display(self.get_res())
