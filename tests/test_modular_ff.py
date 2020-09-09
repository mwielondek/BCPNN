from feedforward_modular import BCPNN
import numpy as np

def clf_factory(test_pattern, targets, module_sizes, normalize=True):
    clf = BCPNN(normalize=normalize)
    clf.fit(test_pattern, targets, module_sizes)
    return clf

def predict_runner(train_pattern, targets, predictions, module_sizes,
            test_pattern=None, mode='proba', atol=0.2, normalize=True):
    clf = clf_factory(train_pattern, targets, module_sizes, normalize)
    if mode == 'proba':
        f = clf.predict_proba
    elif mode == 'log':
        f = clf.predict_log_proba
    elif mode == 'predict':
        f = clf.predict

    if test_pattern is None:
        test_pattern = train_pattern

    output = f(test_pattern)
    assert output.shape == predictions.shape
    # NOTE: below form easier to debug
    # assert (output == predictions).all()
    assert np.allclose(output, predictions, atol=atol)

class TestUnitTests:

    def testIndexTransform(self):
        clf = BCPNN()

        clf.module_sizes = np.array([1])
        modular = (0, 0)
        flat = 0
        assert clf._modular_idx_to_flat(*modular) == flat
        assert clf._flat_to_modular_idx(flat) == modular

        clf.module_sizes = np.array([1, 1])
        modular = (1, 0)
        flat = 1
        assert clf._modular_idx_to_flat(*modular) == flat
        assert clf._flat_to_modular_idx(flat) == modular

        clf.module_sizes = np.array([2, 2])
        modular = (0, 1)
        flat = 1
        assert clf._modular_idx_to_flat(*modular) == flat
        assert clf._flat_to_modular_idx(flat) == modular
        modular = (1, 0)
        flat = 2
        assert clf._modular_idx_to_flat(*modular) == flat
        assert clf._flat_to_modular_idx(flat) == modular

        clf.module_sizes = np.array([1, 3, 2])
        modular = (2, 1)
        flat = 5
        assert clf._modular_idx_to_flat(*modular) == flat
        assert clf._flat_to_modular_idx(flat) == modular


def testModuleSize2_log():
    train_pattern = np.array([[0, 1], [1, 0]])
    targets       = np.array([[0, 1], [1, 0]])
    predictions   = np.array([[np.log(1/4), 0], [0, np.log(1/4)]])
    module_sizes  = np.array([2, 2])
    predict_runner(train_pattern, targets, predictions, module_sizes, mode='log', atol=0.001)

def testModuleSize2_proba_no_norm():
    train_pattern = np.array([[0, 1], [1, 0]])
    targets       = np.array([[0, 1], [1, 0]])
    predictions   = np.array([[0.25, 1], [1, 0.25]])
    module_sizes  = np.array([2, 2])
    predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba', normalize=False, atol=0.001)

def testModuleSize2_proba():
    train_pattern = np.array([[0, 1], [1, 0]])
    targets       = np.array([[0, 1], [1, 0]])
    predictions   = np.array([[0.2, 0.8], [0.8, 0.2]])
    module_sizes  = np.array([2, 2])
    predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba', normalize=True, atol=0.001)
