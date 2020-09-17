import numpy as np
from .encoder import ComplementEncoder

class BCPNN:
    """
    The Bayesian Confidence Propagation Neural Network (BCPNN)

    Roughly following the architecture devised in  "The Use of a
    Bayesian Neural Network Model for Classification Tasks", Anders Holst, 1997.

    @author: M. Wielondek
    """

    def __init__(self, normalize=True, g=1):
        # Whether to use threshold fn or normalize the output in transfer fn
        self.normalize = normalize
        # Controls number of clusters, as per "CLUSTERING OF STORED MEMORIES
        # IN AN ATTRACTOR NETWORK WITH LOCAL COMPETITION", A. Lansner, 2006.
        self.g = g

    def _transformX_enabled(fn):
        """Adds a transformX parameter and subsequent logic. Function needs to take X as first argument"""

        def wrapper(self, X, *args, transformX=False, **kwargs):
            if transformX:
                # if X are not on the complement unit form
                X = ComplementEncoder.transform(X)
            return fn(self, X, *args, **kwargs)

        return wrapper

    @_transformX_enabled
    def fit(self, X, Y, module_sizes=None):
        """Where X is an array of samples and Y is either:

        - an array of probabilities of respective sample belonging to each class
        OR
        - an array of class indices to which each samples belongs to (assumes
         100% probability)

        and module_sizes is an array designating the size of each module (= hypercolumn)
        """

        assert X.shape[0] == Y.shape[0]

        self.X_ = X
        self.n_training_samples, self.n_features_ = X.shape

        if Y.ndim == 1:
            self.classes_ = self._unique_labels(Y)
            self.Y_ = self._class_idx_to_prob(Y)
        else:
            self.Y_ = Y
            self.classes_ = np.arange(Y.shape[1])
        self.n_classes_ = self.classes_.shape[0]

        if module_sizes is None:
            # assume complementary units, ie module size 2 for all X modules
            module_sizes = np.hstack((np.full(self.n_features_ // 2, 2), self.n_classes_))

        assert module_sizes.sum() == self.n_features_ + self.n_classes_
        self.module_sizes = module_sizes
        # How many modules Y consists of
        self.n_modules_y = np.flatnonzero(np.cumsum(self.module_sizes[::-1]) == self.n_classes_)[0] + 1
        self._assert_module_normalization(self.X_)

        # Extending X with y values allows us to work with
        # only one array throughout the code, enabling us to
        # write input/output-layer agnostic functions.
        self.training_activations = np.concatenate((self.X_, self.Y_), axis=1)
        # Necessary padding into X to arrive at the y values
        self.y_pad = self.n_features_

        # Pre-calculate beta and weights
        self.beta = np.array([self._get_beta(self.y_pad + j) for j in self.classes_])

        self.weights = np.zeros((self.n_features_, self.n_classes_))
        for i in range(self.n_features_):
            for j in range(self.n_classes_):
                self.weights[i][j] = self._get_weights(i, j + self.y_pad)

    @_transformX_enabled
    def predict_log_proba(self, X):
        """Classify and return the log probabilities of each sample
        belonging to respective class."""
        self._assert_module_normalization(X)
        beta = self.beta
        n_samples = X.shape[0]
        s = np.zeros((n_samples, self.n_classes_))
        n_modules_j = self.n_modules_y
        n_modules_i = self.module_sizes.size - n_modules_j
        for sample in range(X.shape[0]):
            for sjj in range(self.n_classes_):
                theta = np.zeros(n_modules_i)
                for i in range(theta.shape[0]):
                    ii = self._modular_idx_to_flat(i, 0)   # idx of start of module i
                    theta[i] = self.weights[ii:ii+self.module_sizes[i], sjj].dot(X[sample, ii:ii+self.module_sizes[i]].T)
                sigma_log = np.log(theta).sum()
                s[sample, sjj] = beta[sjj] + sigma_log
        return s

    @_transformX_enabled
    def predict_proba(self, X):
        """Classify and return the probabilities of each sample
        belonging to respective class."""
        return self._transfer_fn(self.predict_log_proba(X))

    @_transformX_enabled
    def predict(self, X):
        """Classify and return the class index of each sample."""
        probabilities = self.predict_proba(X)
        max_probability_class = list(map(np.argmax, probabilities))
        return self.classes_[max_probability_class]

    @_transformX_enabled
    def score(self, X, y):
        """Classify and compare the predicted labels with y, returning
        the mean accuracy."""
        return (self.predict(X) == y).sum() / len(y)

    def _assert_module_normalization(self, X):
        """ Checks that the values in each module sum up to 1"""
        modules = np.split(X, np.cumsum(self.module_sizes[:-self.n_modules_y-1]), axis=1)
        sum_to_one = np.allclose([module.sum(axis=1) for module in modules], 1)
        if not sum_to_one:
            raise self.NormalizationError("values within each module should sum up to 1")

    class NormalizationError(ValueError):
        pass

    def _modular_idx_to_flat(self, i, iprim):
        """ Translates modular index on the form i,i' to flat index"""
        return self.module_sizes[:i].sum() + iprim

    def _flat_to_modular_idx(self, flat_idx):
        module_cumsum = np.hstack((0, np.cumsum(self.module_sizes)))
        i = np.flatnonzero(module_cumsum > flat_idx)[0] - 1
        iprim = flat_idx - module_cumsum[i]
        return (i, iprim)

    def _transfer_fn(self, support):
        # Since the independence assumption often is only approximately
        # fulfilled, these equations give only an approximation of the
        # probability. Therefore the formulas will eventually produce
        # probability estimates larger than 1. To prevent this, one
        # alternative is to use a threshold in the transfer function
        # (Holst 1997, eq. 2.14)
        if not self.normalize:
            return np.exp(np.where(support > 0, 0, support))

        expsup = np.exp(support * self.g)
        # split returns views into existing array so we can work directly with expsup
        # split using cumsum will always return one empty array, hence the :-1
        modules = np.split(expsup, np.cumsum(self.module_sizes[-self.n_modules_y:-1]), axis=1)
        for m in modules:
            module_sz = m.shape[1]
            # sum the module and tile appropriately to enable elementwise division
            total = np.tile(m.sum(axis=1), (module_sz, 1)).T
            m /= total
        return expsup

    @staticmethod
    def _unique_labels(y):
        """Returns a set of labels, also ensuring correct progression."""
        labels = np.array(sorted(set(y)))
        # ensure following progression 0,1,2,... without skipping any values
        assert sum(labels) == sum(range(labels[-1] + 1))
        return labels

    @staticmethod
    def _class_idx_to_prob(y):
        """Receives a 1D array of class indexes and returns a 2D array of the
        shape (n_samples, n_classes) with probability values for each sample
        belogning to given class (setting probability to 100%).
        """
        classes = set(y)

        # make sure each class is represented at least once 0,1,2,...
        assert classes == set(range(max(y) + 1))

        Y = np.zeros(tuple(map(lambda x: len(x), [y, classes])))
        for i, cls_idx in enumerate(y):
            Y[i][cls_idx] = 1
        return Y

    def _get_beta(self, i):
        # log( P(x_i) ) - the bias term
        c = self._get_prob(i)
        # we deal with log(0) case, as per Holst 1997 (eq. 2.37)
        if c == 0:
            return np.log(1 / (self.n_training_samples ** 2))
        return np.log(c)

    def _get_prob(self, i):
        # P(x_i)
        # Check how many times x_i occured,
        # divided by the number of samples
        return self.training_activations[:, i].sum() / self.n_training_samples

    def _get_joint_prob(self, i, j):
        # P(x_i, x_j)
        # Check how many times x_i occured together with x_j,
        # divided by number of samples
        return (self.training_activations[:, i] * self.training_activations[:, j]).sum() / self.n_training_samples

    def _get_weights(self, i, j):
        # P(x_i, x_j) / ( P(x_i) x P(x_j) )
        pi, pj = self._get_prob(i), self._get_prob(j)
        pij = self._get_joint_prob(i, j)
        if pi == 0 or pj == 0:
            return 1
        # we deal with log(0) case, as per Holst 1997 (eq. 2.36)
        if pij == 0:
            return 1 / self.n_training_samples
        return pij / (pi * pj)

    """
     For compatibility with sklearn, below are
     needed to conform to Estimator type.
    """
    def get_params(self, deep=True):
        # BCPNN takes no init arguments
        return {"normalize": self.normalize, "g": self.g}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
