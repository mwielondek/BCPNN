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

    def fit(self, X, y=None):
        X = np.array(X)
        n_features = X.shape[1]
        mod_sz = np.full(n_features, 2)
        # append y modules if given
        if y is not None:
            y_module_size = np.unique(y).size
            mod_sz = np.hstack((mod_sz, y_module_size))
        self.module_sizes_ = mod_sz
        return self

    def transform(self, X):
        """
        Transforms a vector of real values within the range (0, 1) onto complementary units form.

        >>> ComplementEncoder.transform([[0.5, 0.2]])
        array([[0.5, 0.5, 0.2, 0.8]])
        """
        X = np.array(X)
        return np.dstack((X, 1-X)).reshape(X.shape[0], X.shape[1] * 2)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def inverse_transform(X):
        return np.array(X)[:, ::2]

class OneHotEncoder(skEncoder):
    """
    Uses sklearn's OneHotEncoder and returns module sizes for use with fit method. For use with discrete features.
    """
    def __init__(self):
        return super().__init__(sparse=False, dtype='int')

    def fit(self, X, y=None):
        super().fit(X)
        mod_sz = list(map(len, (self.categories_)))
        if 1 in mod_sz:
            # Amend categories; zero variance features (all same value) will have a single category
            # We want to transform it onto binary form, ie each feature should at least have two categories
            # (categories is a standard list of unequal element size, so we can't use numpy's array indexing)
            for i in np.where(np.array(mod_sz) == 1)[0]:
                self.categories_[i] = np.array([0, 1])
            mod_sz = list(map(len, (self.categories_)))
        # append y modules if given
        if y is not None:
            y_module_size = np.unique(y).size
            mod_sz = np.hstack((mod_sz, y_module_size))
        self.module_sizes_ = mod_sz
        return self
