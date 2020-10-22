import numpy as np

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