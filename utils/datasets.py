import numpy as np
import pandas as pd
import warnings

from sklearn.datasets import load_digits

from ..BCPNN.encoder import ComplementEncoder, OneHotEncoder

def transform_patterns(patterns):
    """One hot encode patterns, ie [[0,1,2]] -> [[1, 0, 0, 0, 1, 0, 0, 0, 1]]"""
    n_features = patterns.shape[1]
    n_values = np.unique(patterns).size

    assert np.min(patterns) == 0, "patterns must start with 0"
    assert (np.unique(patterns) == np.arange(n_values)).all(), "patterns must be consequitive 0...n"

    def f(row):
        assert type(row) is np.ndarray
        a = np.zeros(n_values*n_features)
        idx = row + np.arange(row.size) * n_values
        a[idx] = 1
        return a
    return np.apply_along_axis(f, 1, patterns)

def generate_synthetic_dataset(levels=3):
    """Generate a dataset consisting of classes that
    recursively split into 3 subclasses, `levels` deep.
    Idea from Levin 1995"""
    def to_base3(n, pad=levels):
        if not n:
            return '0' * pad
        s = ""
        while n:
            s = str(n % 3) + s
            n //= 3
        return ("{:0"+str(pad)+"d}").format(int(s))

    def from_base3(n):
        l = list(map(int, list(str(n))))
        s = 0
        for i, d in enumerate(l[::-1]):
            s += 3 ** i * d
        return s

    ones = np.ones(levels, dtype=int)
    rows = 3**levels
    base3nums = [to_base3(x) for x in range(rows)]
    base3idx = np.array(list( map(list, base3nums) )).astype(int) + ones
    out = np.zeros((rows, levels))
    for i, idx in enumerate(base3idx):
        idxstr = ''.join(idx.astype(str))
        prefixes = [idxstr[:a] for a in range(1,len(idxstr)+1)]
        out[i] = list(map(from_base3, prefixes))
    return out.astype(int) - ones

def load_zoo(mode='all', transform=False, recurrent=False):
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

    if transform:
        enc = OneHotEncoder()
        X = enc.fit_transform(X, y, recurrent=recurrent)
        return X, y, enc.module_sizes_

    return X, y

def load_mushrooms(transform=False, recurrent=False):
    path = '../../datasets/mushrooms/'
    shrooms = pd.read_csv(path+'agaricus-lepiota.data', header=0, names=np.arange(23)).astype(str)
    X_df = shrooms.drop(columns=0)
    X = X_df.values
    y = shrooms[0].to_numpy()
    # encode labels onto 0,1
    y = np.where(y=='e', 0, 1)

    if transform:
        enc = OneHotEncoder()
        X = enc.fit_transform(X, y, recurrent=recurrent)
        return X, y, enc.module_sizes_

    return X, y

def load_digits_784(res_factor=1, transform=False, recurrent=False):
    mnist = pd.read_csv('parent/../datasets/mnist_784.csv')
    y = mnist['class'].astype(np.int8)
    X = mnist.iloc[:,:-1]
    X = (X.values.astype(np.uint8) / 255).astype(np.float32)
    if res_factor > 1:
        X = X.reshape(X.shape[0], 28, 28)[:, ::res_factor, ::res_factor].reshape(X.shape[0], -1)

    if transform:
        enc = ComplementEncoder()
        X = enc.fit_transform(X, y, recurrent=recurrent)
        return X, y, enc.module_sizes_

    return X,y

def load_digits_64(transform=False, recurrent=False):
    X,y = load_digits(return_X_y=True)
    X /= 16

    if transform:
        enc = ComplementEncoder()
        X = enc.fit_transform(X, y, recurrent=recurrent)
        return X, y, enc.module_sizes_

    return X,y

def stratified_split(X, y, n=10):
    """Return a subset of X and y with n samples of each class"""
    classes, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()
    if min_samples < n:
        warnings.warn(("At least one class consists of only {} samples. Adjusting n for uniform"
        + " percentage of samples for each class.").format(min_samples))
        n = min_samples
    idx = np.array([np.flatnonzero(y == cls)[:n] for cls in classes]).ravel()
    return X[idx], y[idx]
