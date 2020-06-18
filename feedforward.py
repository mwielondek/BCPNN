import numpy as np

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

    def fit(self, X, Y):
        """Where X is an array of samples and Y is either:

        - an array of probabilities of respective sample belonging to each class
        OR
        - an array of class indices to which each samples belongs to (assumes
         100% probability)
        """

        assert X.shape[0] == Y.shape[0]

        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape

        if Y.ndim == 1:
            self.classes_ = self._unique_labels(Y)
            self.Y_ = self._class_idx_to_prob(Y)
        else:
            self.Y_ = Y
            self.classes_ = np.arange(Y.shape[1])
        self.n_classes_ = self.classes_.shape[0]

        # Extending X with y values allows us to work with
        # only one array throughout the code, enabling us to
        # write input/output-layer agnostic functions.
        self.X_ = np.concatenate((self.X_, self.Y_), axis=1)
        # Necessary padding into X to arrive at the y values
        self.y_pad = self.n_features_

    def predict_log_proba(self, X):
        """Classify and return the log probabilities of each sample
        belonging to respective class."""
        n_samples, n_features = X.shape

        support = np.empty((n_samples, self.n_classes_))
        for sample_idx, x in enumerate(X):
            for cls_idx, j in enumerate(self.classes_):
                j = self.y_pad + j

                beta = self._get_beta(j)

                weights = 0
                for i in range(n_features):
                    weights += self._get_weights(j, i) * x[i]

                h = beta + weights
                support[sample_idx][cls_idx] = h

        return support

    def predict_proba(self, X):
        """Classify and return the probabilities of each sample
        belonging to respective class."""
        return self._transfer_fn(self.predict_log_proba(X))

    def predict(self, X):
        """Classify and return the class index of each sample."""
        probabilities = self.predict_proba(X)
        max_probability_class = list(map(np.argmax, probabilities))
        return self.classes_[max_probability_class]

    def score(self, X, y):
        """Classify and compare the predicted labels with y, returning
        the mean accuracy."""
        return (self.predict(X) == y).sum() / len(y)

    def _transfer_fn(self, support):
        # Since the independence assumption often is only approximately
        # fulfilled, these equations give only an approximation of the
        # probability. Therefore the formulas will eventually produce
        # probability estimates larger than 1. To prevent this, one
        # alternative is to use a threshold in the transfer function
        # (Holst 1997, eq. 2.14)
        if not self.normalize:
            return np.exp(np.where(support > 0, 0, support))

        # Or we can normalize the output over the hypercolumn (eq 2.15).
        expsup = np.exp(support * self.g)
        for sample_idx, sample in enumerate(expsup):
            # DEBUG: remove assert in final version
            sample_sum = sample.sum()
            assert sample_sum > 0
            expsup[sample_idx] /= sample_sum
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
            return np.log(1 / (self.n_samples_ ** 2))
        return np.log(c)

    def _get_prob(self, i):
        # P(x_i)
        # Check how many times x_i occured,
        # divided by the number of samples
        return self.X_[:, i].sum() / self.n_samples_

    def _get_joint_prob(self, i, j):
        # P(x_i, x_j)
        # Check how many times x_i occured together with x_j,
        # divided by number of samples
        return (self.X_[:, i] * self.X_[:, j]).sum() / self.n_samples_

    def _get_weights(self, i, j):
        # P(x_i, x_j) / ( P(x_i) x P(x_j) )

        ci, cj = self._get_prob(i), self._get_prob(j)
        cij = self._get_joint_prob(i, j)
        if ci == 0 or cj == 0:
            return 0
        # we deal with log(0) case, as per Holst 1997 (eq. 2.36)
        if cij == 0:
            return np.log(1 / self.n_samples_)
        return np.log( cij / (ci * cj) )

    """
     For compatibility with sklearn, below are
     needed to conform to Estimator type.
    """
    def get_params(self, deep=True):
        # BCPNN takes no init arguments
        return {"normalize": self.normalize, "G-value": self.G}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
