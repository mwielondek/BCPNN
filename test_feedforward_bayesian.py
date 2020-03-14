from feedforward_bayesian import BCPNN
import numpy as np


test_pattern = np.array([
    [1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1]
    ])
targets = np.array([0, 1])

def clf_factory(test_pattern=test_pattern, targets=targets):
    clf = BCPNN()
    clf.fit(test_pattern, targets)
    return clf

def test_predict():
    clf = clf_factory()
    f = clf.predict

    assert (f(test_pattern) == [0, 1]).all()

    one_flipped = test_pattern[:]
    one_flipped[0][0] = 0
    one_flipped[1][4] = 1
    assert (f(one_flipped) == [0, 1]).all()


def test_predict_proba():
    test_pattern = np.array([
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 0]
        ])

    clf = clf_factory(test_pattern)
    f = clf.predict_proba

    predictions = np.array([[1, 0], [0, 1]])
    assert (f(test_pattern) == predictions).all()

def test_predict_proba2():
    test_pattern = np.array([
        [1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1]
        ])

    clf = clf_factory(test_pattern)
    f = clf.predict_proba

    predictions = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert (f(test_pattern) == predictions).all()
