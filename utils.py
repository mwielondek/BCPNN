import numpy as np
import matplotlib.pyplot as plt

def count_unique_patterns(X, decimals=None):
    """Returns distinct pattern count, rounding up to `decimals`"""
    if decimals:
        X = X.round(decimals=decimals)
    return np.unique(X, axis=0).shape[0]

def get_cluster_arrays(X, oneindexed=False, decimals=None):
    """Returns an array of clusters, where each value corresponds to sample ID"""
    clusters = {}
    if decimals:
        X = X.round(decimals=decimals)
    for pat in np.unique(X, axis=0):
        clusters[pat.tostring()] = []
    for idx,pat in enumerate(X):
        clusters[pat.tostring()].append(idx + oneindexed)
    return list(clusters.values())

def get_cluster_ids(X, decimals=None):
    """Returns an array of cluster IDs, where the index corresponds to sample ID"""
    n_samples, _ = X.shape
    clusters = get_cluster_arrays(X, False, decimals)
    arr = np.zeros(n_samples).astype(int)
    for cidx, c in enumerate(clusters):
        for sample in c:
            arr[sample] = cidx
    return arr

def collect_cluster_ids(clf, X, gvals, decimals=None, fit_params=None, predict_params=None):
    """Get cluster IDs as a function of g values"""
    n_samples, _ = X.shape
    n_gvals = len(gvals)
    clusters = np.empty((n_gvals, n_samples))

    clf.fit(X, **fit_params)

    clf.g = gvals[0]
    pred = clf.predict(X, **predict_params)
    clusters[0] = get_cluster_ids(pred, decimals=decimals)

    for idx, g in list(enumerate(gvals))[1:]:
        clf.g = g
        pred = clf.predict(X, **predict_params)
        prev = clusters[idx-1]
        current = get_cluster_ids(pred, decimals=decimals)
        clusters[idx] = current

    return clusters.astype(int)

def get_distance(X):
    """Calculate hamming distance between patterns"""
    dists = {}
    upper = X.shape[1]
    for idx, pat in enumerate(X):
        dists[idx+1] = {}
        for idx2, pat2 in enumerate(X):
            dists[idx+1][idx2+1] = upper - (pat == pat2).sum()


    table = {}
    for n in range(upper+1):
        table[n] = {}
    for idx, v in dists.items():
        for n in range(upper+1):
            keys = { key for key,value in v.items() if value == n }
            table[n][idx] = keys
    return scrub_dict(table)

# removes empty values
def scrub_dict(d):
    if type(d) is dict:
        return dict((k, scrub_dict(v)) for k, v in d.items() if v and scrub_dict(v))
    else:
        return d

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

### MNIST
def dispmnist(x, ax=None, shape=(28,28)):
    """Display MNIST figure"""
    if ax is None:
        ax = plt
    ax.imshow(x.reshape(shape), cmap='gray_r')

def dispcompare(x, y, title=None):
    """Display MNIST vectors vertically stacked above each other"""
    n_samples, _ = x.shape
    fig, axs= plt.subplots(2, n_samples, figsize=(2*n_samples, 1.5*2))
    if title:
        fig.suptitle(title)
    [axi.set_axis_off() for axi in axs.ravel()]
    for i, row in enumerate(axs.T):
        dispmnist(x[i], row[0])
        dispmnist(y[i], row[1])

def addsalt(X, prob=0.1):
    """Add salt to pattern with `prob` probability by using the complement"""
    xr = X.flatten()
    for i,_ in enumerate(xr):
        if np.random.choice([True, False], p=[prob, 1-prob]):
            xr[i] = max(0.8, 1 - xr[i])
    return xr.reshape(X.shape)
