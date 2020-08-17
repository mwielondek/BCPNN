import numpy as np
from utils import opt_cluster_labeling as ocl
from utils import get_mapping as gm

""" A trick to introduce shorthand _[..] for np.array([])"""
class ShorthandArray(object):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return np.array(key)
        else:
            return np.array([key])

_ = ShorthandArray()

class TestLabeling:
    """
    Legend:
        c = num clusters in current
        p = num clusters in prev

    Case A: c = p
    Case B: c < p
    Case C: c > p
    """

    def test_case_a(self):
        p = _[0, 1]
        c = _[1, 2]
        r = _[0, 1]
        assert np.all(ocl(p, c) == r)

        p = _[0, 1]
        c = _[3, 2]
        r = _[0, 1]
        assert np.all(ocl(p, c) == r)

        p = _[0, 1, 3]
        c = _[1, 2, 3]
        r = _[0, 1, 3]
        assert np.all(ocl(p, c) == r)

        p = _[0, 1, 2]
        c = _[1, 2, 0]
        r = _[0, 1, 2]
        assert np.all(ocl(p, c) == r)

        p = _[3, 3, 3]
        c = _[1, 1, 1]
        r = _[3, 3, 3]
        assert np.all(ocl(p, c) == r)


    def test_case_b(self):
        p = _[0, 1, 3]
        c = _[1, 1, 2]
        r = _[0, 0, 3]
        assert np.all(ocl(p, c) == r)

        p = _[3, 3, 3]
        c = _[0, 1, 2]
        r = _[3, 0, 1]
        assert np.all(ocl(p, c) == r)

        p = _[0, 1, 2]
        c = _[1, 1, 1]
        r = _[0, 0, 0]
        assert np.all(ocl(p, c) == r)

        p = _[0, 1, 2, 3]
        c = _[1, 2, 0, 0]
        r = _[0, 1, 2, 2]
        assert np.all(ocl(p, c) == r)

    def test_case_c(self):
        p = _[0, 1, 3]
        c = _[1, 1, 2]
        r = _[0, 0, 3]
        assert np.all(ocl(p, c) == r)

        p = _[1, 0, 1]
        c = _[0, 1, 2]
        r = _[1, 0, 2]
        assert np.all(ocl(p, c) == r)

        p = _[0, 0, 0]
        c = _[0, 1, 2]
        r = _[0, 1, 2]
        assert np.all(ocl(p, c) == r)

        p = _[1, 1, 1]
        c = _[0, 1, 2]
        r = _[1, 0, 2]
        assert np.all(ocl(p, c) == r)

        p = _[4, 5, 4, 4, 5]
        c = _[0, 0, 1, 1, 2]
        r = _[5, 5, 4, 4, 0]
        assert np.all(ocl(p, c) == r)

        p = _[4, 4, 4, 4, 4]
        c = _[0, 0, 1, 1, 1]
        r = _[0, 0, 4, 4, 4]
        assert np.all(ocl(p, c) == r)

        p = _[4, 4, 4, 4, 4, 2]
        c = _[0, 0, 1, 1, 1, 0]
        r = _[2, 2, 4, 4, 4, 2]
        assert np.all(ocl(p, c) == r)

        p = _[4, 4, 4, 4, 4, 2, 3]
        c = _[0, 0, 1, 1, 1, 0, 0]
        r = _[2, 2, 4, 4, 4, 2, 2]
        assert np.all(ocl(p, c) == r)

class TestLabelingSub:

    def test_get_mapping(self):
        a = _[0, 0, 1, 2]
        b = _[4, 4, 5, 5]
        r = _[([0, 4], 2), ([1, 5], 1), ([2, 5], 1)]
        np.testing.assert_array_equal(gm(a,b), r)
