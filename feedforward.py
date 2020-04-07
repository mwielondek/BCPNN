import numpy as np

class BCPNN:

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]

        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape

        self.y_ = y
        self.classes_ = self._unique_labels(y)
        self.n_classes_ = self.classes_.shape[0]

        # extend X with y values
        extension = np.zeros((self.n_samples_, self.n_classes_))
        # necessary padding into X to arrive at the y values
        self.y_pad = self.n_features_

        for i, cls in enumerate(y):
            extension[i][cls] = 1

        self.X_ = np.concatenate((self.X_, extension), axis=1)

    def predict_log_proba(self, X):
        n_samples, n_features = X.shape

        support = np.empty((n_samples, self.n_classes_))
        for sample_idx, x in enumerate(X):
            for cls_idx, j in enumerate(self.classes_):
                # pad to arrive at y values in X - see fit method
                j = self.y_pad + j

                beta = self._get_beta(j)

                weights = 0
                for i in range(n_features):
                    weights += self._get_weights(j, i) * x[i]

                h = beta + weights
                support[sample_idx][cls_idx] = h

        return support

    def predict_proba(self, X):
        return self._transfer_fn(self.predict_log_proba(X))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        max_probability_class = list(map(np.argmax, probabilities))
        return self.classes_[max_probability_class]

    def _transfer_fn(self, support):
        # Since the independence assumption often is only approximately
        # fulfilled, these equations give only an approximation of the
        # probability. Therefore the formulas will eventually produce
        # probability estimates larger than 1. To prevent this, one
        # alternative is to use a threshold in the transfer function
        # - Holst 1997 (eq. 2.14)
        return np.exp(np.where(support > 0, 0, support))

    @staticmethod
    def _unique_labels(y):
        labels = np.array(sorted(set(y)))
        # check we didn't skip any values, ie it must follow 0,1,2,3,...
        assert sum(labels) == sum(range(labels[-1] + 1))
        return labels

    def _get_beta(self, i):
        # log( P( x_i ) )

        # we deal with log(0) case, as per Holst 1997 (eq. 2.37)
        c = self._get_prob(i)
        if c == 0:
            return np.log(1 / (self.n_samples_ ** 2))
        return np.log(c)

    def _get_prob(self, i):
        # check how many times x_i occured divided by the number of samples
        return self.X_[:, i].sum() / self.n_samples_

    def _get_joint_prob(self, i, j):
        # check how many times x_i occured together with x_j
        # the divided by number of samples
        return (self.X_[:, i] * self.X_[:, j]).sum() / self.n_samples_

    def _get_weights(self, i, j):
        # P(x_i, x_j) / ( P(x_i) x P(x_j) )

        # we deal with log(0) case, as per Holst 1997 (eq. 2.36)
        ci, cj = self._get_prob(i), self._get_prob(j)
        cij = self._get_joint_prob(i, j)
        if ci == 0 or cj == 0:
            return 0
        if cij == 0:
            return np.log(1 / self.n_samples_)
        return np.log( cij / (ci * cj) )
