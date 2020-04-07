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
        y = [3,4,7,2,2,3]
        assert (BCPNN._unique_labels(y) == [2,3,4,7]).all()

    def test_transfer_fn(self):
        g = self.clf._transfer_fn
        f = lambda s: g(np.array(s))

        support = [0, 0]
        assert (f(support) == [1, 1]).all()
        support = np.log([1, 2, 4, 1])
        assert (f(support) == [1, 1, 1, 1]).all()
        # test 2d arrays
        support = [[0, 0], [0, 0]]
        assert (f(support) == [[1, 1], [1, 1]]).all()
        support = np.log([[1, 2], [4, 1]])
        assert (f(support) == [[1, 1], [1, 1]]).all()

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

    def predict_runner(self, test_pattern, targets, predictions,
                        mode='proba', atol=0.4):
        clf = self.clf_factory(test_pattern, targets)
        if mode == 'proba':
            f = clf.predict_proba
        elif mode == 'log':
            f = clf.predict_log_proba
        elif mode == 'predict':
            f = clf.predict

        output = f(test_pattern)
        assert output.shape == predictions.shape
        # NOTE: below form easier to debug
        # assert (output == predictions).all()
        assert np.allclose(output, predictions, atol=atol)

    def test_basic(self):
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
        test_pattern = np.array([
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 1]
            ])
        targets = np.array([0, 0, 1])

        # predict_proba
        predictions = np.array([[1, 0], [1, 1], [0, 1]])
        self.predict_runner(test_pattern, targets, predictions, mode='proba')
        # predict
        predictions = np.array([0, 0, 1])
        self.predict_runner(test_pattern, targets, predictions, mode='predict')

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

    # def test_different_sizes3(self):
    #     # Test different size of n_features and n_classes
    #     # n_features = 3, n_classes = 3
    #     test_pattern = np.array([
    #         [1, 0, 1],
    #         [1, 0, 1],
    #         [0, 1, 0]
    #         ])
    #     targets = np.array([0, 1, 2])
    #
    #     # predict_proba
    #     predictions = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    #     self.predict_runner(test_pattern, targets, predictions, mode='proba')
    #     # predict
    #     predictions = np.array([0, 0, 1])
    #     self.predict_runner(test_pattern, targets, predictions, mode='predict')
    #
    # def test_different_sizes4(self):
    #     # Test different size of n_features and n_classes
    #     # n_features = 4, n_classes = 3
    #     test_pattern = np.array([
    #         [1, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 1, 1],
    #         [0, 1, 0, 1, 0, 0, 0]
    #         ])
    #     targets = np.array([0, 1, 2])
    #
    #     # predict_proba
    #     predictions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #     self.predict_runner(test_pattern, targets, predictions, mode='proba')
    #     # predict
    #     predictions = np.array([0, 1, 2])
    #     self.predict_runner(test_pattern, targets, predictions, mode='predict')
