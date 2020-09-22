import pytest
import numpy as np

from BCPNN.feedforward_modular import BCPNN
from BCPNN.encoder import ComplementEncoder

## TEST UTILS
def clf_factory(test_pattern, targets, module_sizes, normalize=True):
    clf = BCPNN(normalize=normalize)
    clf.fit(test_pattern, targets, module_sizes)
    return clf

def predict_runner(train_pattern, targets, predictions, module_sizes,
            test_pattern=None, mode='proba', atol=0.001, normalize=True):
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

    def testComplementEncoder(self):
        X = np.array([[0.5, 0.2], [1, 0]])
        Xt = ComplementEncoder().fit_transform(X)
        assert np.array_equal(Xt, np.array([[0.5, 0.5, 0.2, 0.8], [1, 0, 0, 1]]))
        assert np.array_equal(ComplementEncoder.inverse_transform(Xt), X)


    @pytest.fixture(scope="function")
    def clf(self):
        return BCPNN()

    def testIndexTransform(self, clf):
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

    class TestTransformX:

        def testEmptyModuleSize(self, clf):
            X = np.array([[0, 1]])
            y = np.array([0])
            clf.fit(X, y, transformX=False)
            assert np.array_equal(clf.X_, X)
            assert np.array_equal(clf.module_sizes, [2, 1])

        def testTransformX0(self, clf):
            X = np.array([[0, 1]])
            y = np.array([0])
            clf.fit(X, y, transformX=True)
            assert np.array_equal(clf.X_, [[1,0, 0,1]])
            assert np.array_equal(clf.module_sizes, [2, 2, 1])

        def testTransformX1(self, clf):
            # One discrete feature
            X = np.array([[1], [2], [3]])
            y = np.array([0, 1, 2])
            clf.fit(X, y, transformX=True)
            X_pred = np.array([ [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1] ])
            assert np.array_equal(clf.X_, X_pred)
            assert np.array_equal(clf.module_sizes, [3, 3])

        def testTransformX2(self, clf):
            # One binary feature
            X = np.array([[1], [1], [2]])
            y = np.array([0, 1, 2])
            clf.fit(X, y, transformX=True)
            X_pred = np.array([[1, 0], [1, 0], [0, 1]])
            assert np.array_equal(clf.X_, X_pred)
            assert np.array_equal(clf.module_sizes, [2, 3])

        def testTransformX3(self, clf):
            # One binary, one discrete
            X = np.array([[1, 1], [1, 2], [2, 3]])
            y = np.array([0, 0, 0])
            clf.fit(X, y, transformX=True)
            X_pred = np.array([ [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1] ])
            assert np.array_equal(clf.X_, X_pred)
            assert np.array_equal(clf.module_sizes, [2, 3, 1])


    class TestNormalization:

        def testCheckNormalization1(self, clf):
            X = np.array([[0, 1]])
            module_sizes = np.array([2, 2])
            clf.fit(X, X, module_sizes)

            X = np.array([[1, 1]])
            with pytest.raises(BCPNN.NormalizationError):
                clf.fit(X, X, module_sizes)

        def testCheckNormalization2(self, clf):
            X = np.array([[0, 1], [1/2, 1/2]])
            module_sizes = np.array([2, 2])
            clf.fit(X, X, module_sizes)
            clf._assert_module_normalization(X)

            X = np.array([[1, 1], [1/2, 1/2]])
            module_sizes = np.array([2, 2])
            with pytest.raises(BCPNN.NormalizationError):
                clf.fit(X, X, module_sizes)

        def testCheckNormalization3(self, clf):
            X = np.array([[0, 1/2, 1/2, 1/2, 1/2]])
            module_sizes = np.array([3, 2, 2])
            clf.fit(X, X[:, :2], module_sizes)

            X = np.array([[1, 1, 1, 1, 1]])
            module_sizes = np.array([3, 2, 2])
            with pytest.raises(BCPNN.NormalizationError):
                clf.fit(X, X[:, :2], module_sizes)

        def testCheckNormalization4(self, clf):
            # No module sizes given
            X = np.array([[0, 1]])
            clf.fit(X, X)

            X = np.array([[1, 1]])
            with pytest.raises(BCPNN.NormalizationError):
                clf.fit(X, X)

class TestModule:

    class TestBinary:

        def testModuleSize2_2_log(self):
            train_pattern = np.array([[0, 1], [1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[np.log(1/4), 0], [0, np.log(1/4)]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

        def testModuleSize2_2_proba_no_norm(self):
            train_pattern = np.array([[0, 1], [1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[0.25, 1], [1, 0.25]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba', normalize=False)

        def testModuleSize2_2_proba(self):
            train_pattern = np.array([[0, 1], [1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[0.2, 0.8], [0.8, 0.2]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba')

        def testModuleSize3_2_proba(self):
            train_pattern = np.array([[0, 1, 0], [1, 0, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[0.2, 0.8], [0.8, 0.2]])
            module_sizes  = np.array([3, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba')

        def testModuleSize3_3_log(self):
            train_pattern = np.array([[0, 1, 0], [1, 0, 0]])
            targets       = np.array([[0, 1, 0], [1, 0, 0]])
            predictions   = np.array([[np.log(1/4), 0, np.log(1/4)], [0, np.log(1/4), np.log(1/4)]])
            module_sizes  = np.array([3, 3])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

        def testModuleSize3_3_proba(self):
            train_pattern = np.array([[0, 1, 0], [1, 0, 0]])
            targets       = np.array([[0, 1, 0], [1, 0, 0]])
            predictions   = np.array([[1/6, 2/3, 1/6], [2/3, 1/6, 1/6]])
            module_sizes  = np.array([3, 3])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba')

        def testModuleSize2_2_2_log(self):
            train_pattern = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[np.log(1/8), np.log(2)], [np.log(2), np.log(1/8)]])
            module_sizes  = np.array([2, 2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

        def testModuleSize2_2_2_proba(self):
            train_pattern = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[1/17, 16/17], [16/17, 1/17]])
            module_sizes  = np.array([2, 2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='proba')

        def testModuleSize2_3_2_log(self):
            train_pattern = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[np.log(1/8), np.log(2)], [np.log(2), np.log(1/8)]])
            module_sizes  = np.array([2, 3, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

        def testModuleNormalizationAssertion1_train(self):
            train_pattern = np.array([[1, 1]])
            targets       = np.array([[1, 0]])
            test_pattern  = np.array([[0, 0]])
            predictions   = np.array([[1, 0]])
            module_sizes  = np.array([2, 2])
            with pytest.raises(BCPNN.NormalizationError):
                predict_runner(train_pattern, targets, predictions, module_sizes, test_pattern=test_pattern,
                 mode='proba')

        def testModuleNormalizationAssertion1_test(self):
            train_pattern = np.array([[1, 0]])
            targets       = np.array([[1, 0]])
            test_pattern  = np.array([[1, 1]])
            predictions   = np.array([[1, 0]])
            module_sizes  = np.array([2, 2])
            with pytest.raises(BCPNN.NormalizationError):
                predict_runner(train_pattern, targets, predictions, module_sizes, test_pattern=test_pattern,
                 mode='proba')

    class TestFractional:

        def testModuleSize2_2_log(self):
            train_pattern = np.array([[1/3, 2/3], [2/3, 1/3]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[np.log(4/9), np.log(5/9)], [np.log(5/9), np.log(4/9)]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

        def testModuleSize2_3_2_log(self):
            train_pattern = np.array([[1/3, 2/3, 1/4, 1/4, 2/4], [1, 0, 0, 1, 0]])
            targets       = np.array([[0, 1], [1, 0]])
            predictions   = np.array([[np.log(155/480), np.log(288/240)], [np.log(6/5), np.log(1/10)]])
            module_sizes  = np.array([2, 3, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, mode='log')

    class TestDifferentTestPattern:

        def testModuleSize2_2_log(self):
            train_pattern = np.array([[1, 0], [0, 1]])
            targets       = np.array([[1, 0], [0, 1]])
            test_pattern  = np.array([[1/2, 1/2], [2/3, 1/3]])
            predictions   = np.array([[np.log(5/8), np.log(5/8)], [np.log(3/4), np.log(1/2)]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, test_pattern=test_pattern, mode='log')

        def testModuleSize2_2_single_log(self):
            train_pattern = np.array([[1, 0]])
            targets       = np.array([[1, 0]])
            test_pattern  = np.array([[1, 0], [2/3, 1/3]])
            predictions   = np.array([[0, 0], [0, 0]])
            module_sizes  = np.array([2, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, test_pattern=test_pattern, mode='log')

        def testModuleSize3_2_log(self):
            train_pattern = np.array([[1, 0, 0], [0, 1, 0]])
            targets       = np.array([[1, 0], [0, 1]])
            test_pattern  = np.array([[0, 0, 1], [0, 1/2, 1/2]])
            predictions   = np.array([[np.log(1/2), np.log(1/2)], [np.log(3/8), np.log(3/4)]])
            module_sizes  = np.array([3, 2])
            predict_runner(train_pattern, targets, predictions, module_sizes, test_pattern=test_pattern, mode='log')
