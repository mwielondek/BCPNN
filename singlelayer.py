import numpy as np

class BinvecOneHotEncoder:

    @staticmethod
    def transform(X):
        """
        Transforms a vector of binary patterns onto the one hot similar form
        where each individual binary value x_ij represents whether value x_i
        in the original vector was 1 or 0, where j = {1, 0}.
        """
        X_new = np.empty((X.shape[0], X.shape[1]*2))
        for i, pattern in enumerate(X):
            for j, attr in enumerate(pattern):
                jj = j * 2
                if attr == 1:
                    X_new[i][jj:jj+2] = [1, 0]
                else:
                    X_new[i][jj:jj+2] = [0, 1]
        return X_new

    @staticmethod
    def inverse_transform(X):
        X_inv = np.empty((X.shape[0], X.shape[1]//2))
        for i, pattern in enumerate(X):
            X_inv[i] = pattern[::2]
        return X_inv


class BCPNN:

    N_VALUES = 2

    def fit(self, X):
        self.X_ = X
        self.n_samples_, self.n_features_ = X.shape

    def update(self, X):
        activations = np.empty_like(X)
        for i, _ in enumerate(activations):

            beta = self._get_beta(i)

            #        sigma = sum_i (log ( sum_i' ( w_ij * P(x_i) ) ))
            # over hypercolumns ^         ^ over neurons in each hypercolumn
            sigma = 0
            for j in range(0, activations.shape[0], self.N_VALUES):
                inner_sum = 0
                for k in range(self.N_VALUES):
                    inner_sum += self._get_weights(j+k, i) * self._get_prob(j)
                sigma += np.log(inner_sum)

            h = beta + sigma

            activations[i] = np.exp(h)

        # normalize output in each hypercolumn
        return self._normalize(activations)


    def retrieve(self, X):
        Y = np.empty_like(X)
        for i, pattern in enumerate(X):

            y = np.empty_like(pattern)
            old = pattern

            while not np.allclose(old, y):
                old = y
                y = self.update(y)

            Y[i] = y

        return Y

    def _get_beta(self, i):
        # log( P( x_i ) )

        # we deal with log(0) case, as per Holst 1997
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

        # we deal with log(0) case, as per Holst 1997
        ci, cj = self._get_prob(i), self._get_prob(j)
        cij = self._get_joint_prob(i, j)
        if ci == 0 or cj == 0:
            return 0
        if cij == 0:
            return np.log(1 / self.n_samples_)
        return cij / (ci * cj)

    def _normalize(self, activations):
        # divide by sum_j( e^h_j ) where j are all the possible values of h

        for i in range(0, activations.shape[0], self.N_VALUES):

            denominator = sum(map(np.exp, activations[i:i+self.N_VALUES]))

            for k in range(self.N_VALUES):
                activations[i+k] = activations[i+k] / denominator

        return activations
