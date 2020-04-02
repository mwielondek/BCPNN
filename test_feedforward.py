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
        f = self.clf._transfer_fn

        support = [0, 0]
        assert (f(support) == [0.5, 0.5]).all()
        support = np.log([1, 2, 4, 1])
        assert (f(support) == [1/3, 2/3, 4/5, 1/5]).all()
        # test 2d arrays
        support = [[0, 0], [0, 0]]
        assert (f(support) == [[0.5, 0.5], [0.5, 0.5]]).all()
        support = np.log([[1, 2], [4, 1]])
        assert (f(support) == [[1/3, 2/3], [4/5, 1/5]]).all()


test_pattern = np.array([
    [1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1]
    ])
targets = np.array([0, 1])

def clf_factory(test_pattern=test_pattern, targets=targets):
    clf = BCPNN()
    clf.fit(test_pattern, targets)
    return clf

def predict_proba_runner(test_pattern, targets, predictions):
    clf = clf_factory(test_pattern, targets)
    f = clf.predict_proba
    output = f(test_pattern)
    assert output.shape == predictions.shape
    assert (output == predictions).all()

# def test_predict_proba():
#     # Test different size fo n_features and n_classes
#     test_pattern = np.array([
#         [1, 0, 1, 0],
#         [1, 0, 1, 0],
#         [1, 0, 1, 0]
#         ])
#     targets = np.array([0, 2, 1])
#     predictions = np.array([[1, 0], [1, 0], [1, 0]])
#     predict_proba_runner(test_pattern, targets, predictions)
#
# def test_predict_proba():
#     test_pattern = np.array([
#         [1, 0, 1, 0, 0, 1],
#         [1, 0, 0.5, 0.5, 0, 1],
#         [0.7, 0.3, 1, 0, 0, 1],
#         [0, 1, 0, 1, 1, 0],
#         [0.9, 0.1, 0.9, 0.1, 0.1, 0.9]
#         ])
#     targets = np.array([0, 0, 0, 1, 1])
#
#     clf = clf_factory(test_pattern, targets)
#     f = clf.predict_proba
#
#     output = f(test_pattern)
#     predictions = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]])
#     assert output.shape == predictions.shape
#     assert (output == predictions).all()
#
# def test_predict():
#     clf = clf_factory()
#     f = clf.predict
#
#     assert (f(test_pattern) == targets).all()
#
#     # one_flipped = test_pattern[:]
#     # one_flipped[0][0] = 0
#     # one_flipped[1][4] = 1
#     # assert (f(one_flipped) == [0, 1]).all()
