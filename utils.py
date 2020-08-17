import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from recurrent import rBCPNN

def opt_cluster_labeling(prev, current):
    """Try to maximize number of elements that don't change clusters."""

    # if already matched, return
    if (current == prev).sum() == current.size:
        return current

    unique_curr = np.unique(current)  # unique clusters
    n_unique_curr = unique_curr.size
    target_clusters = np.unique(prev) # only interested in translating into cluster id from prev
    n_unique_prev = target_clusters.size

    # check if the same but translated
    if n_unique_prev == n_unique_curr:
        idx1, idx2 = np.unique(current, return_index=True)  # idx1 = unique cls, idx2 = at pos
        idx3 = prev[idx2]                                   # cls in prev at pos of unique els in current
        if np.unique(idx3).size == idx3.size:               # all must be unique, otherwise no 1-1 mapping
            # translate and compare
            proto = np.zeros(idx1.max() + 1)
            proto[idx1] = idx3
            if (proto[current] == prev).all():
                return prev

    # start with clusters that has the most corresponding entries
    mappings = get_mapping(current, prev)[:, 0]         # already sorted, remove counts
    mappings = np.array(mappings.tolist())
    placeholder = -1                                    # for unassigned targets
    translations = np.ones(n_unique_curr).astype(int) * placeholder
    while mappings.size:
        ffrom, to = mappings[0]
        translations[unique_curr == ffrom] = to
        # remove from mappings all where from or to are the same
        idx1, = (mappings[:,0] == ffrom).nonzero()
        idx2, = (mappings[:,1] == to).nonzero()
        mappings = np.delete(mappings, np.append(idx1, idx2), axis=0)

    # check which from nodes we don't have a translation for and update with lowest available id
    idx = (translations == placeholder)
    used_ids = translations[(translations >= 0) & (translations < n_unique_curr)]
    lowest = np.delete(list(range(n_unique_curr)), used_ids)
    translations[idx] = lowest[:np.flatnonzero(idx).size]

    # map current using unique elements table
    translated_current = list(map(lambda x: np.where(unique_curr == x)[0][0], current))
    return translations[translated_current]

def get_mapping(x, y):
    """Returns list of mappings from x to y sorted by counts."""
    mappings = np.vstack((x, y)).T
    mappings = np.unique(mappings, axis=0, return_counts=True)
    mappings = map(np.ndarray.tolist, mappings)
    mappings = list(zip(*mappings))
    return np.array(sorted(mappings, key=itemgetter(1), reverse=True))


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

def collect_cluster_ids(X, gvals, decimals=None, clf=None, optimize=True):
    """Get cluster IDs as a function of g values"""
    n_samples, _ = X.shape
    n_gvals = len(gvals)
    clusters = np.empty((n_gvals, n_samples))

    if clf is None:
        # TODO find optimal values for tol and max_iter
        clf = rBCPNN(normalize=True, tol=1e-15, max_iter=1e8, g=1)
    clf.fit(X)

    clf.g = gvals[0]
    pred = clf.predict(X)
    clusters[0] = get_cluster_ids(pred, decimals=decimals)

    for idx, g in list(enumerate(gvals))[1:]:
        clf.g = g
        pred = clf.predict(X)
        prev = clusters[idx-1]
        current = get_cluster_ids(pred, decimals=decimals)
        if optimize:
            clusters[idx] = opt_cluster_labeling(prev, current)
        else:
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

def draw_clustergram(gvals, clusters, targets=None, legend=True, cluster_padding=0.3, one_indexed=False):
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (8,7)

    n_gvals, n_samples = clusters.shape
    padding = np.linspace(-cluster_padding, cluster_padding, n_samples)

    fig, ax = plt.subplots()
    ax.plot(gvals, clusters + padding + 1, 'o-', markerfacecolor=(1,1,1,0.9), markersize=3, drawstyle='steps-mid')

    if targets is not None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        assert max(targets) < len(colors), "not enough colors for 1-1 mapping with targets"
        for i, line in enumerate(ax.lines[:-1]):
            line.set_color(colors[targets[i]])

    n_clusters = list(map(lambda row: np.unique(row).size, clusters))
    ax.plot(gvals, n_clusters, ':', color='#CCCCCC')

    ax.grid(which='both', axis='x', color='#CCCCCC', linestyle=(0, (1, 10)))
    ax.set_xticks(gvals, minor=True)
    ax.set_xticks(gvals[::n_gvals//4], minor=False)

    yticks = np.arange(1, max(n_clusters)+1, 1)
    ax.set_yticks(yticks, minor=False)
    yticks = map(lambda x: [x - padding[0], x + padding[0]], yticks)
    yticks = np.array(list(yticks)).flatten().tolist()
    ax.set_yticks(yticks, minor=True)

    ax.set_ylabel('Cluster ID')
    ax.set_xlabel('g-values')
    if legend:
        if targets is None:
            ax.legend((np.arange(n_samples) + one_indexed).tolist() + ['# of clu.'], loc=0)
        else:
            vals, idx = np.unique(targets, return_index=True)
            nplines = np.array(ax.lines)
            ax.legend(nplines[idx], vals + one_indexed, loc=0)

    fig.patch.set_facecolor('xkcd:mint green')

    for idx, _ in list(enumerate(yticks))[::2]:
        ax.axhspan(*yticks[idx:idx+2], alpha=0.1)

    return ax

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
