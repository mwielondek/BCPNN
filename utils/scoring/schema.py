## Loader fns

import pandas as pd
import numpy as np
from sklearn import datasets
import warnings

from . import Scorer

def load_zoo(mode='all'):
    path = '../../datasets/zoo-animal-classification/'
    zoo = pd.read_csv(path+'zoo.csv')
    zoo

    # remove class type (target) and name
    X_df = zoo.drop(['animal_name', 'class_type'], axis=1)

    if mode == 'binary':
        # remove class type and legs to only keep binary attributes
        X_df = X_df.drop(['legs'], axis=1)

    X = X_df.values

    y = zoo['class_type'].to_numpy() - 1 # to make it zero indexed

    return X, y

def load_mushrooms():
    path = '../../datasets/mushrooms/'
    shrooms = pd.read_csv(path+'agaricus-lepiota.data', header=0, names=np.arange(23)).astype(str)
    X_df = shrooms.drop(columns=0)
    X = X_df.values
    y = shrooms[0].to_numpy()
    # encode labels onto 0,1
    y = np.where(y=='e', 0, 1)
    return X, y

## Preprocess fns

from BCPNN.encoder import OneHotEncoder, ComplementEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import TransformerMixin, BaseEstimator

def onehot_encode(X, y, handle_unknown='ignore'):
    enc = OneHotEncoder(handle_unknown=handle_unknown)
    Xt = enc.fit_transform(X, y)
    return Xt, y, enc.module_sizes_

def discretize_onehot_encode(X, y):
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    return onehot_encode(kbd.fit_transform(X), y)

class UniformScaler(BaseEstimator, TransformerMixin):
    """Scales all features by the same factor"""

    def __init__(self, factor=None):
        self.factor = factor

    def fit(self, X, y=None, **fit_params):
        if self.factor is None:
            self.factor = 1 / X.max()
        return self

    def transform(self, X, **transform_params):
        return X * self.factor

def uniform_scale_complement_encode(X, y):
    Xt = UniformScaler().fit_transform(X)
    enc = ComplementEncoder()
    Xt = enc.fit_transform(Xt, y)
    return Xt, y, enc.module_sizes_

## Schema
def run_default():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        run_schema(scoring_schema)


def run_schema(schema, clfidx=None):

    def pprint(msg):
        print("****** {:^103} ******".format(str(msg)))

    sc = Scorer(clfidx)

    data = schema['data']
    for dtype,dtype_val in data.items():
        print('\n'*2)
        pprint(dtype.upper())

        for dataset in dtype_val['data']:
            pprint(dataset['name'].capitalize())
            X, y = dataset['loader'](**dataset['loader_params'])

            for preprocess in dtype_val['preprocess']:
                if callable(preprocess):
                    pprint("Preprocess fn: {}".format(preprocess.__name__))
                    Xt, y, modsz = preprocess(X, y)
                    scores = sc.score_all(Xt, y, method='cv', fit_params=dict(mBCPNN__module_sizes=modsz))
                else:
                    pprint("Pipeline preprocess: {}".format(preprocess))
                    scores = sc.score_all(X, y, pipeline__preprocess=preprocess)
                sc.pretty_print(scores)
                print()
            print()


scoring_schema = {
    'data': {
        'continuous': {
            'data': [{
                'name': 'wine',
                'loader': datasets.load_wine,
                'loader_params': dict(return_X_y=True)
            }, {
                'name': 'iris',
                'loader': datasets.load_iris,
                'loader_params': dict(return_X_y=True)
            }],
            'preprocess': [['discretize', 'onehot_encode'], discretize_onehot_encode]
        },
        'discrete': {
            'data': [{
                'name': 'mnist',
                'loader': datasets.load_digits,
                'loader_params': dict(return_X_y=True)
            }, {
                'name': 'zoo',
                'loader': load_zoo,
                'loader_params': dict(mode='all')
            }],
            'preprocess': [['scale','complement_encode'], discretize_onehot_encode, uniform_scale_complement_encode]
        },
        'binary': {
            'data': [{
                'name': 'zoo',
                'loader': load_zoo,
                'loader_params': dict(mode='binary')
            }],
            'preprocess': [['complement_encode'],[]]
        },
        'categorical': {
            'data': [{
                'name': 'mushrooms',
                'loader': load_mushrooms,
                'loader_params': dict()
            }],
            'preprocess': [onehot_encode]
        },
    }
}
