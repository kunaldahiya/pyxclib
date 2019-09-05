from sklearn.preprocessing import normalize as scale
import numpy as np
import _pickle as pickle
from . import data_utils
import os


class LabelsBase(object):
    """Base class for Labels
    Parameters
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or np.ndarray or None, optional, default=None
        data is already provided
    """

    def __init__(self, data_dir, fname, Y=None, sp_format='csc'):
        self.sp_format = sp_format
        self.Y = self.load(data_dir, fname, Y)
        self.adjust_sparse_format()

    def adjust_sparse_format(self):
        if self._valid:
            self.Y = self.Y.asformat(self.sp_format)

    def _select_instances(self, indices):
        return self.Y[indices] if self._valid else None

    def _select_labels(self, indices):
        return self.Y[:, indices] if self._valid else None

    def normalize(self, norm='max', copy=False):
        self.Y = scale(self.Y, copy=copy, norm=norm) if self._valid else None

    def load(self, data_dir, fname, Y):
        if Y is not None:
            return Y
        elif fname is None:
            return None
        else:
            fname = os.path.join(data_dir, fname)
            if fname.lower().endswith('.pkl'):
                return pickle.load(open(fname, 'rb'))['Y']
            elif fname.lower().endswith('.txt'):
                return data_utils.read_sparse_file(
                    fname, dtype=np.float32)
            else:
                raise NotImplementedError("Unknown file extension")

    def get_invalid(self, axis=0):
        return np.where(self.frequency(axis) == 0)[0] if self._valid else None

    def get_valid(self, axis=0):
        return np.where(self.frequency(axis) > 0)[0] if self._valid else None

    def remove_invalid(self, axis=0):
        indices = self.get_valid(axis)
        self.Y = self.index_select(indices)
        return indices

    def binarize(self):
        if self._valid:
            self.Y.data[:] = 1.0

    def index_select(self, indices, axis=1):
        """
            Choose only selected labels or instances
        """
        if axis == 0:
            return self._select_instances(indices)
        elif axis == 1:
            return self._select_labels(indices)
        else:
            NotImplementedError("Unknown Axis.")

    def frequency(self, axis=0):
        return np.array(self.Y.astype(np.bool).sum(axis=axis)).ravel() \
            if self._valid else None

    def transpose(self):
        return self.Y.transpose() if self._valid else None

    @property
    def _valid(self):
        return self.Y is not None

    @property
    def num_instances(self):
        return self.Y.shape[0] if self._valid else -1

    @property
    def num_labels(self):
        return self.Y.shape[1] if self._valid else -1

    @property
    def shape(self):
        return (self.num_instances, self.num_labels)

    def __getitem__(self, index):
        return self.Y[index] if self._valid else None

    @property
    def data(self):
        return self.Y

    @data.setter
    def data(self, _Y):
        self.Y = _Y
        self.adjust_sparse_format()
