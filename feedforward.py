import numpy as np

class BCPNN:

    # Num possible values for each feature
    N_VALUES = 2

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

    def predict_proba(self, X):
        n_samples, n_features = X.shape

        support = np.empty((n_samples, self.n_classes_))
        for sample_idx, x in enumerate(X):
            for cls_idx, j in enumerate(self.classes_):
                beta = self._get_beta(j)

                # from Sandberg 2002 (equation 3):
                #        sigma = sum_i (log ( sum_i' ( w_ij * P_X(x_i) ) ))
                # over hypercolumns ^         ^ over neurons in each hypercolumn
                outer_sum = 0
                for i in range(0, n_features, self.N_VALUES):
                    inner_sum = 0
                    for k in range(self.N_VALUES):
                        prod = self._get_weights(i + k, j)
                        prod *= x[i + k]
                        inner_sum += prod
                    outer_sum += np.log(inner_sum)

                h = beta + outer_sum
                support[sample_idx][cls_idx] = h

        return self._transfer_fn(support)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        max_probability_class = list(map(np.argmax, probabilities))
        return self.classes_[max_probability_class]

    def _transfer_fn(self, support):
        # calculate the exponential and normalize output in each hypercolumn
        # From Sandberg 2002 (eq 6):
        #     e^hjj' / sum_j' ( e^hjj' )
        activations = np.exp(support).T

        for i in range(0, activations.shape[0], self.N_VALUES):
            denominator = sum(activations[i:i + self.N_VALUES])

            for k in range(self.N_VALUES):
                activations[i + k] /= denominator

        return activations.T

    @staticmethod
    def _unique_labels(y):
        return np.array(sorted(set(y)))

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
        return cij / (ci * cj)
