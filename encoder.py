import numpy as np

class BinvecOneHotEncoder:

    @staticmethod
    def transform(X):
        """
        Transforms a vector of binary patterns onto the one hot similar form
        where each individual binary value x_ij represents whether value x_i
        in the original vector was 1 or 0, where j = {1, 0}.
        """
        X = np.array(X)
        X_new = np.empty((X.shape[0], X.shape[1]*2))
        for i, pattern in enumerate(X):
            for j, attr in enumerate(pattern):
                jj = j * 2
                if attr == 1:
                    X_new[i][jj:jj+2] = [1, 0]
                else:
                    X_new[i][jj:jj+2] = [0, 1]
        return X_new.astype(int)

    @staticmethod
    def inverse_transform(X):
        X_inv = np.empty((X.shape[0], X.shape[1]//2))
        for i, pattern in enumerate(X):
            X_inv[i] = pattern[::2]
        return X_inv
