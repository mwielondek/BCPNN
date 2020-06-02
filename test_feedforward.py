from feedforward import BCPNN
from encoder import BinvecOneHotEncoder as encoder
import numpy as np


class TestUnitTests:
    def setup(self):
        self.clf = BCPNN()

        self.test_pattern = np.array([
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0]
            ])
        self.transformed_pattern = encoder.transform(self.test_pattern)
        self.y = np.array([0,1])

        self.clf.fit(self.test_pattern, self.y)


    def test_encoder_encode(self):
        transformed = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        assert (self.transformed_pattern[0] == transformed).all()

    def test_encoder_decode(self):
        inverse = encoder.inverse_transform(self.transformed_pattern)
        assert (inverse == self.test_pattern).all()

    def test_get_prob(self):
        f = self.clf._get_prob

        assert f(0) == 1
        assert f(1) == 0.5
        assert f(2) == 0.5
        assert f(3) == 1
        assert f(4) == 0

    def test_get_joint_prob(self):
        f = self.clf._get_joint_prob

        assert f(0, 0) == 1
        assert f(0, 1) == 0.5
        assert f(1, 4) == 0

    def test_get_prob_for_y(self):
        f = self.clf._get_prob

        assert f(self.clf.y_pad + 0) == 0.5
        assert f(self.clf.y_pad + 1) == 0.5

    def test_get_joint_prob_for_y(self):
        f = self.clf._get_joint_prob

        assert f(0, self.clf.y_pad + 0) == 0.5
        assert f(0, self.clf.y_pad + 1) == 0.5
        assert f(1, self.clf.y_pad + 1) == 0.5
        assert f(1, self.clf.y_pad + 0) == 0
        assert f(4, self.clf.y_pad + 1) == 0

    def test_unique_label(self):
        y = [3,4,1,5,6,7,2,2,3]
        assert (BCPNN._unique_labels(y) == [1,2,3,4,5,6,7]).all()

    def test_class_idx_to_prob(self):
        y = np.arange(3)
        prediction = np.eye(3)
        assert (BCPNN._class_idx_to_prob(y) == prediction).all()

        y = np.array([1, 2, 0])
        prediction = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        assert (BCPNN._class_idx_to_prob(y) == prediction).all()

        y = np.array([1, 1, 0])
        prediction = np.array([[0, 1], [0,1], [1, 0]])
        assert (BCPNN._class_idx_to_prob(y) == prediction).all()

    def test_transfer_fn(self):
        g = self.clf._transfer_fn
        f = lambda s: g(np.array(s))

        support = [0, 0]
        assert (f(support) == [1, 1]).all()
        support = np.log([1, 2, 4, 1])
        assert (f(support) == [1, 1, 1, 1]).all()
        # test 2d arrays
        support = [[0, 0], [0, 0]]
        assert (f(support) == [[0.5, 0.5], [0.5, 0.5]]).all()
        support = np.log([[2, 6], [4, 1]]).tolist()
        assert (f(support) == np.array([[0.25, 0.75], [0.8, 0.2]])).all()

class TestProba:

    test_pattern = np.array([
        [1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1]
        ])
    targets = np.array([0, 1])

    def clf_factory(self, test_pattern=test_pattern, targets=targets):
        clf = BCPNN()
        clf.fit(test_pattern, targets)
        return clf

    def predict_runner(self, train_pattern, targets, predictions,
                test_pattern=None, mode='proba', atol=0.2):
        clf = self.clf_factory(train_pattern, targets)
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

    def test_basic1(self):
        test_pattern = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
            ])
        targets = np.array([0, 1])

        # predict_log_proba
        predictions = np.array([[ 0.69314718, -2.07944154],
                                [-2.07944154,  0.69314718]])
        self.predict_runner(test_pattern, targets, predictions,
                                mode='log', atol=0)
        # predict_proba
        predictions = np.array([[1, 0], [0, 1]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 1])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')

    def test_basic2(self):
        train_pattern = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
            ])
        targets = np.array([0, 1])

        test_pattern = np.append(train_pattern, [[1, 1, 1, 1]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba')
        # predict
        predictions = np.array([0, 1, 0])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

    def test_basic3(self):
        train_pattern = np.array([
            [1, 0, 1],
            [0, 1, 0]
            ])
        targets = np.array([0, 1])

        test_pattern = np.append(train_pattern, [[1, 1, 1], [1, 1, 0]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0], [0, 1], [0.8, 0.2], [0.5, 0.5]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba')
        # predict
        predictions = np.array([0, 1, 0, 0])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

    def test_basic3_with_transform(self):
        train_pattern = np.array([
            [1, 0, 1],
            [0, 1, 0]
            ])
        targets = np.array([0, 1])

        test_pattern = np.append(train_pattern, [[1, 1, 1], [1, 1, 0]], axis=0)

        train_pattern = encoder.transform(train_pattern)
        test_pattern = encoder.transform(test_pattern)

        # predict_proba
        predictions = np.array([[1, 0], [0, 1], [0.7, 0.3], [0.3, 0.7]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba')
        # predict
        predictions = np.array([0, 1, 0, 1])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

    def test_basic4(self):
        train_pattern = np.array([
            [1, 0, 0],
            [0, 1, 0]
            ])
        targets = np.array([0, 1])

        # TEST CASE
        test_pattern = np.append(train_pattern, [[0, 0, 1]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba', atol=0.25)
        # predict
        predictions = np.array([0, 1, 0])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

        # TEST CASE
        test_pattern = np.append(train_pattern, [[0, 0, 0]], axis=0)
        # predict_proba
        predictions = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba', atol=0.25)
        # predict
        predictions = np.array([0, 1, 0])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

    def test_basic5(self):
        train_pattern = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ])
        targets = np.array([0, 1, 2])

        # TEST CASE
        test_pattern = np.append(train_pattern, [[0, 0, 0]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba', atol=0.2)
        # predict
        predictions = np.array([0, 1, 2, 0])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='predict')

        # TEST CASE
        test_pattern = np.append(train_pattern, [[1, 1, 1]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.3, 0.3]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba', atol=0.2)

        # TEST CASE
        test_pattern = np.append(train_pattern, [[1, 0, 1]], axis=0)

        # predict_proba
        predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0, 0.3]])
        self.predict_runner(train_pattern, targets, predictions,
                    test_pattern=test_pattern, mode='proba', atol=0.2)

    def test_different_sizes1(self):
        # Test different size of n_features and n_classes
        # n_features = 5, n_classes = 2
        test_pattern = np.array([
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]
            ])
        targets = np.array([0, 0, 1])

        # predict_proba
        predictions = np.array([[1, 0], [1, 0], [0, 1]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 0, 1])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')

    def test_different_sizes2(self):
        # Test different size of n_features and n_classes
        # n_features = 4, n_classes = 3
        test_pattern = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ])
        targets = np.array([0, 1, 2])

        # predict_proba
        predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 1, 2])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')

    def test_different_sizes3(self):
        # Test different size of n_features and n_classes
        # n_features = 3, n_classes = 3
        test_pattern = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [0, 1, 0]
            ])
        targets = np.array([0, 2, 1])

        # predict_proba
        predictions = np.array([[0.5, 0, 0.5], [0.5, 0, 0.5], [0, 1, 0]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 0, 1])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')

    def test_different_sizes4(self):
        # Test different size of n_features and n_classes
        # n_features = 4, n_classes = 3
        test_pattern = np.array([
            [1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 0]
            ])
        targets = np.array([0, 1, 2])

        # predict_proba
        predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 1, 2])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')
