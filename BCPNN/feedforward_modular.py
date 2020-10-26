import numpy as np
from functools import reduce
from .encoder import OneHotEncoder, ComplementEncoder

class BCPNN:
    """
    The Bayesian Confidence Propagation Neural Network (BCPNN)

    Roughly following the architecture devised in  "The Use of a
    Bayesian Neural Network Model for Classification Tasks", Anders Holst, 1997.

    @author: M. Wielondek
    """

    def __repr__(self):
        return "mffBCPNN()"

    def __init__(self, normalize=True, g=1, encoder='onehot'):
        # Whether to use threshold fn or normalize the output in transfer fn
        self.normalize = normalize
        # Controls number of clusters, as per "CLUSTERING OF STORED MEMORIES
        # IN AN ATTRACTOR NETWORK WITH LOCAL COMPETITION", A. Lansner, 2006.
        self.g = g
        # Pick OneHotEncoder for discertely valued features, or ComplementEncoder
        # for when the features are continous and represent probabilities.
        self.encoder = {'onehot': OneHotEncoder(), 'complement': ComplementEncoder()}[encoder]

    def _transformX_enabled(fn):
        """Adds a transformX parameter and subsequent logic. Function needs to take X as first argument"""

        def wrapper(self, X, *args, transformX=False, **kwargs):
            if transformX:
                # if X are not on the complement unit form
                X = self.encoder.transform(X)
            return fn(self, X, *args, **kwargs)

        return wrapper

    def fit(self, X, y, module_sizes=None, transformX=False):
        """Where X is an array of samples and y is either:

        - an array of probabilities of respective sample belonging to each class
        OR
        - an array of class indices to which each samples belongs to (assumes
         100% probability)

        and module_sizes is an array designating the size of each module (= hypercolumn)
        """

        assert X.shape[0] == y.shape[0]

        if y.ndim == 1:
            self.classes_ = self._unique_labels(y)
            self.Y_ = self._class_idx_to_prob(y)
        else:
            self.Y_ = y
            self.classes_ = np.arange(y.shape[1])
        self.n_classes_ = self.classes_.shape[0]

        if transformX:
            X = self.encoder.fit_transform(X, self.classes_)
            module_sizes = self.encoder.module_sizes_

        self.X_ = X
        self.n_training_samples, self.n_features_ = X.shape

        if module_sizes is None:
            # assume complementary units, ie module size 2 for all X modules
            module_sizes = np.hstack((np.full(self.n_features_ // 2, 2), self.n_classes_))
        else:
            module_sizes = self._get_value(module_sizes)

        assert module_sizes.sum() == self.n_features_ + self.n_classes_, "wrong dim of module_sizes"
        self.module_sizes = module_sizes
        # How many modules y consists of
        self.n_modules_y = np.flatnonzero(np.cumsum(self.module_sizes[::-1]) == self.n_classes_)[0] + 1
        self.n_modules_x = self.module_sizes.size - self.n_modules_y
        self._assert_module_normalization(self.X_)
        self.x_module_sections = np.cumsum(self.module_sizes[:-self.n_modules_y-1])
        self.y_module_sections = np.cumsum(self.module_sizes[-self.n_modules_y:-1])

        # Extending X with y values allows us to work with
        # only one array throughout the code, enabling us to
        # write input/output-layer agnostic functions.
        self.training_activations = np.concatenate((self.X_, self.Y_), axis=1)
        # Necessary padding into X to arrive at the y values
        self.y_pad = self.n_features_

        # Pre-calculate probability tables, beta, and weights
        self.joint_prob = np.dot(self.X_.T, self.Y_) / self.n_training_samples
        self.prob = self.training_activations.sum(axis=0) / self.n_training_samples
        self.beta = self._get_beta()
        self.weights = self._get_weights()
        self.weight_modules = np.split(self.weights.T, self.x_module_sections, axis=1)

    @_transformX_enabled
    def predict_log_proba(self, X, assert_off=False):
        """Classify and return the log probabilities of each sample
        belonging to respective class."""
        if not assert_off:
            self._assert_module_normalization(X)
        beta = self.beta # of shape n_classes_
        # split weights and input into modules
        x_modules = np.split(X, self.x_module_sections, axis=1)
        w_x_modules = zip(self.weight_modules, x_modules)
        # reduce all modules onto a logged dot product of the weights and inputs
        def f(acc, modules_w_x):
            w, x = modules_w_x
            wx = w.dot(x.T)
            return acc + np.log(wx.T)
        outer_sum = reduce(f, w_x_modules, 0)
        return beta + outer_sum

    @_transformX_enabled
    def predict_proba(self, X, **kwargs):
        """Classify and return the probabilities of each sample
        belonging to respective class."""
        return self._transfer_fn(self.predict_log_proba(X, **kwargs))

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

    def _get_value(self, val):
        """Unpack value from function or return value as is if not callable"""
        return val() if hasattr(val, '__call__') else val

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
        modules = np.split(expsup, self.y_module_sections, axis=1)
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
        assert classes == set(range(max(y) + 1)), "classes need to be continuous and zero indexed"

        Y = np.zeros(tuple(map(lambda x: len(x), [y, classes])))
        for i, cls_idx in enumerate(y):
            Y[i][cls_idx] = 1
        return Y

    def _get_beta(self):
        # log( P(x_i) ) - the bias term
        prob_classes = self.prob[self.y_pad:]

        # we deal with log(0) case, as per Holst 1997 (eq. 2.37)
        beta_zero = np.log(1 / (self.n_training_samples ** 2))

        # we expect probabilities for some classes can be zero
        with np.errstate(divide='ignore'):
            log_prob_classes = np.log(prob_classes)

        return np.where(prob_classes == 0, beta_zero, log_prob_classes)

    def _get_weights(self):
        p_mult  = self.prob[:self.y_pad, None] * self.prob[self.y_pad:]
        p_joint = np.where(self.joint_prob == 0, p_mult / self.n_training_samples, self.joint_prob)
        p_mult  = np.where(p_mult == 0, p_joint, p_mult)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = p_joint / p_mult
        res[np.isnan(res)] = 1
        return res


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
