import numpy as np


class BinvecOneHotEncoder:

    def transform(X):
        """
        Transforms a vector of binary patterns onto the one hot similar form
        where each individual binary value x_ij represents whether value x_i
        in the original vector was 1 or 0, where j = {1, 0}.
        """
        X_new = np.empty((X.shape[0],X.shape[1]*2))
        for i, pattern in enumerate(X):
            for j, attr in enumerate(pattern):
                jj = j * 2
                if attr == 1:
                    X_new[i][jj:jj+2] = [1, 0]
                else:
                    X_new[i][jj:jj+2] = [0, 1]
        return X_new


class BCPNN:

    def fit(self, X):
        ...

    def update(self, X):
        ...

    def retrieve(self, X):
        ...
