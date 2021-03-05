from sklearn.preprocessing import normalize as scale
import numpy as np
import _pickle as pickle
import os
from . import data_utils


class FeaturesBase(object):
    """Class for dense features
    Parameters
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: np.ndarray or csr_matrix or None, optional, default=None
        data is already provided
    """

    def __init__(self, data_dir, fname, X=None):
        self.X = self.load(data_dir, fname, X)

    def frequency(self, axis=0):
        return np.array(self.X.sum(axis=axis)).ravel()

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis) == 0)[0]

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis) > 0)[0]

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.index_select(indices)
        return indices

    def _select_instances(self, indices):
        self.X = self.X[indices]

    def _select_features(self, indices):
        # Not valid in general case
        pass

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        # TODO: Load and select from file
        if axis == 0:
            self._select_instances(indices)
        elif axis == 1:
            self._select_features(indices)
        else:
            raise NotImplementedError("Unknown Axis.")

    def load(self, data_dir, fname, X):
        if X is not None:
            return X
        else:
            raise NotImplementedError("Loading module not implemented")

    def normalize(self, norm='l2', copy=False):
        pass

    @property
    def num_instances(self):
        return len(self.X)

    @property
    def num_features(self):
        return self.X.shape[1]

    @property
    def data(self):
        return self.X

    @property
    def shape(self):
        return (self.num_instances, self.num_features)

    def __getitem__(self, index):
        return self.X[index]


class DenseFeatures(FeaturesBase):
    """Class for dense features

    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: np.ndarray or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    """

    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)
        if normalize:
            self.normalize()

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()

    def load(self, data_dir, fname, X):
        if X is not None:
            return super().load(data_dir, fname, X)
        else:
            assert fname is not None, "Filename can not be None."
            fname = os.path.join(data_dir, fname)
            return data_utils.read_gen_dense(fname)


class SparseFeatures(FeaturesBase):
    """Class for sparse features
    
    Arguments
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    X: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the data or not
    """

    def __init__(self, data_dir, fname, X=None, normalize=False):
        super().__init__(data_dir, fname, X)
        if normalize:
            self.normalize()

    def load(self, data_dir, fname, X):
        if X is not None:
            return super().load(data_dir, fname, X)
        else:
            assert fname is not None, "Filename can not be None."
            fname = os.path.join(data_dir, fname)
            return data_utils.read_gen_sparse(fname)

    def normalize(self, norm='l2', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def _select_features(self, indices):
        self.X = self.X[:, indices]

    def frequency(self, axis=0):
        return np.array(self.X.astype(np.bool).sum(axis=axis)).ravel()

    def __getitem__(self, index):
        x = self.X[index].indices
        w = self.X[index].data
        return x, w

    @property
    def num_instances(self):
        return self.X.shape[0]
