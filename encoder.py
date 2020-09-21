import numpy as np
from sklearn.preprocessing import OneHotEncoder as skEncoder

class BinvecOneHotEncoder:
    """
    Encodes data onto the format:
    1 -> [1, 0]
    0 -> [0, 1]
    unknown -> [0, 0]

    This allows for specifying which data is available to us; whereas previously
    we were only able to indicate true values (1) or abscence of data (0), we
    are now able to indicate false known values ([0, 1]) and distinguish those
    from absence of data ([0, 0]).
    """

    @staticmethod
    def transform(X):
        """
        Transforms a vector of binary patterns onto the one hot similar form,
        translating each binary value x_i into separate binary values x_ia and
        x_ib, each representing whether x_i was true or false respectively.

        >>> BinvecOneHotEncoder.transform([[1, 0]])
        array([[1, 0, 0, 1]])
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

class ComplementEncoder:
    """
    Encodes data onto the complement form:
    x -> [x, 1-x]
    """

    @staticmethod
    def transform(X):
        """
        Transforms a vector of real values within the range (0, 1) onto complementary units form.

        >>> ComplementEncoder.transform([[0.5, 0.2]])
        array([[0.5, 0.5, 0.2, 0.8]])
        """
        X = np.array(X)
        return np.dstack((X, 1-X)).reshape(X.shape[0], X.shape[1] * 2)

    @staticmethod
    def inverse_transform(X):
        return np.array(X)[:, ::2]

class OneHotEncoder:
    """
    Uses sklearn's OneHotEncoder and returns module sizes for use with fit method. For use with discrete features.
    """

    @staticmethod
    def transform(X, y=None):
        encoder = skEncoder(sparse=False)
        X_t = encoder.fit_transform(X)
        mod_sz = list(map(len, (encoder.categories_)))
        # append y modules if given
        if y is not None:
            y_module_size = np.unique(y).size
            mod_sz = np.hstack((mod_sz, y_module_size))

        return (X_t, mod_sz)
