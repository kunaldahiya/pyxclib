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

    def normalize(self, norm='max', copy=False):
        self.X = scale(self.X, copy=copy, norm=norm)

    def _select_instances(self, indices):
        return self.X[indices]

    def _select_features(self, indices):
        return self.X[:, indices]

    def index_select(self, indices, axis=1, fname=None):
        """
            Choose only selected labels or instances
        """
        # TODO: Load and select from file
        if axis == 0:
            return self._select_instances(indices)
        else:
            raise NotImplementedError("Unknown Axis.")

    def load(self, data_dir, fname, X):
        if X is not None:
            return X
        else:
            assert fname is not None, "Filename can not be None."
            fname = os.path.join(data_dir, fname)
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['X']
            elif fname.lower().endswith('.npy'):
                return np.load(fname)
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(
                    fname, dtype=np.float32)
            else:
                raise NotImplementedError("Unknown file extension")

    @property
    def data(self):
        return self.X

    @property
    def num_instances(self):
        return self.X.shape[0]

    @property
    def num_features(self):
        return self.X.shape[1]

    @property
    def shape(self):
        return (self.num_instances, self.num_features)

    def __getitem__(self, index):
        return self.X[index]
