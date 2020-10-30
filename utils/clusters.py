import numpy as np

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

def get_cluster_ids(X, decimals=2):
    """Returns an array of cluster IDs, where the index corresponds to sample ID"""
    n_samples, _ = X.shape
    clusters = get_cluster_arrays(X, False, decimals)
    arr = np.zeros(n_samples).astype(int)
    for cidx, c in enumerate(clusters):
        for sample in c:
            arr[sample] = cidx
    return arr

def collect_cluster_ids(clf, X, gvals, decimals=2, fit_params=None, predict_params=None):
    """Get cluster IDs as a function of g values"""
    n_samples, _ = X.shape
    n_gvals = len(gvals)
    clusters = np.empty((n_gvals, n_samples))

    # check if already fitted, otherwise fit
    if not hasattr(clf, 'X_'):
        clf.fit(X, **fit_params)

    if decimals <= 0:
        # default to a precision of `decimals` less SF than clf.TOL
        decimals =  int(np.log10(clf.TOL) * -1) + decimals

    for idx, g in enumerate(gvals):
        clf.g = g
        pred = clf.predict(X, **predict_params)
        clusters[idx] = get_cluster_ids(pred, decimals=decimals)

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
