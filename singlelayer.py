import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from functools import partial

### TODO
#

### Necessary
# predict takes in X = (n_samples, n_features)
# benchmark against zoo

### Improvements
# change to class_count_ and feature_count_ like sklearn BernoulliNB
# caching counts
# hardcoded n and m (for non binary look into LabelBinarizer sklearn)
# gain clarity and refactor get_counts as to make j-notation less confusing
#
# add predict_log_proba and predict_proba
# add score function
#
# change into matrix mult instead of for loops

class BCPNN(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = y.shape[0]
        self.n_samples_, self.n_features_ = X.shape

        self.X_ = X
        self.y_ = y

        # Set alpha to some small value, here 1/C is used (Holst 1997)
        self.alpha = 1 / self.n_samples_

        return self

    def predict(self, X):
        check_is_fitted(self, attributes=None)

        X = check_array(X)
        n_samples = X.shape[0]

        # pi = the conditional probability of the outcome yjj' of the variable Yj given observations in A.
        # calc support value s_j = beta_j + Sum_i(w_ji * o_i)

        # TODO check if works without explicit o_i "filtering factor" - we do need it!
        s_values = np.empty((n_samples, self.n_classes_))
        # TODO can change this into matrix calculation or 2d map? Both for loops.
        for i, x in enumerate(X):
            for j, cls in enumerate(self.classes_):
                beta = self._get_beta(cls)

                observed_features = np.flatnonzero(x)
                sigma = sum(map(partial(self._get_weights, j = cls), observed_features))

                s_values[i][j] = beta + sigma

        highest_prob = list(map(np.argmax, s_values))
        return self.classes_[highest_prob]

    def _get_beta(self, i):
        """beta_i = log  (c_i + alpha / n_i) / (C + alpha) where
        C = count of patterns
        c_i = count of train patterns belonging to class i
        n_i = number of outcomes of Y_i (2 - yes/no?)
        """

        # TODO check n_i correct definition

        assert np.issubdtype(type(i), np.integer)

        c = self.n_samples_
        ci = self._get_counts(j = i)
        ni = 2 # TODO hardcoded
        a = self.alpha

        b = (ci + a / ni) / (c + a)

        return np.log(b)

    def _get_counts(self, i = None, j = None):
        """Number of times that X_i (and X_j if j is set) appear (jointly) in the training set.

        i denotes X_i
        j denotes Y_i

        Note that if X_i is denoting a class these are equivalent:
        c_i = count of training inputs belonging to class i
        c_i = count of times X_i appear in the training set

        c_ij implies P(Y_i, X_j) meaning the count of training samples with
        feature X_j belonging to class i.
        """

        assert i is not None or j is not None

        # to help with caching always order the args
#         if j and i > j:
#             return self._get_counts(j, i)

        counter = 0

        if i is not None and j is not None:
            for idx, x in enumerate(self.X_):
                counter += x[i] and self.y_[idx] == j

        elif j is not None:
            counter = (self.y_ == j).sum()

        else:
            for x in self.X_:
                counter += x[i]

        # TODO cache this to improve perf
        return counter

    def _get_weights(self, i, j):
        """Calculate the weights between node i and j = w_ij.

        j implies Y_i, ie class variable
        """

        c = self.n_samples_
        ci, cj, cij = [self._get_counts(i), self._get_counts(j = j), self._get_counts(i, j)]
        ni, mj = [2, 2] # number of attributes, TODO change from hardcoded
        a = self.alpha

        w = (cij + a / (ni * mj)) * (c + a) / ((ci + a / ni) * (cj + a / mj))

        return np.log(w)
