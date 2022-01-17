import numpy as np
from .data_utils import read_gen_sparse
from ..utils.sparse import normalize, binarize
import os


class LabelsBase(object):
    """Base class for Labels

    Arguments:
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or np.ndarray or None, optional, default=None
        data is already provided
    """

    def __init__(self, data_dir, fname, Y=None, _format='csr'):
        self._format = _format
        self.Y = self.load(data_dir, fname, Y)
        self._adjust_format()

    def _adjust_format(self):
        """
        Modify format of sparse matrix (eg. csr/csc etc.)
        """
        if self._valid:
            self.Y = self.Y.asformat(self._format)

    def instance_select(self, indices):
        """
        Return instances (rows) as per given indices
        """
        return self.Y[indices] if self._valid else None

    def label_select(self, indices):
        """
        Return labels (columns) as per given indices
        """
        return self.Y[:, indices] if self._valid else None

    def normalize(self, norm='max', copy=False):
        self.Y = normalize(self.Y, copy=copy, norm=norm) if self._valid else None

    def load(self, data_dir, fname, Y):
        if Y is not None:
            return Y
        elif fname is None:
            return None
        else:
            fname = os.path.join(data_dir, fname)
            return read_gen_sparse(fname)

    def get_invalid_indices(self, axis=0):
        """
        Get indices of invalid rows/columns (no data available)
        """
        return np.where(self.frequency(axis) == 0)[0] if self._valid else None

    def get_valid_indices(self, axis=0):
        """
        Get indices of only valid rows/columns (data available)
        """
        return np.where(self.frequency(axis) > 0)[0] if self._valid else None

    def _index_select(self, indices, axis=0):
        """
            Keep only given labels or instances (will modify data)
        """
        self.Y = self.index_select(indices, axis=axis)

    def remove_invalid(self, axis=0):
        """
        Get indices of invalid row (axis=0) or column (axis=1)
        Remove them in-place and return indices
        """
        indices = self.get_valid_indices(axis)
        self.Y = self.index_select(indices, axis=1-axis)
        return indices

    def binarize(self):
        if self._valid:
            self.Y = binarize(self.Y, copy=False)

    def index_select(self, indices, axis=1):
        """
            Return data as per given indices and axis
        """
        if axis == 0:
            return self.instance_select(indices)
        elif axis == 1:
            return self.label_select(indices)
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

    @property
    def data(self):
        return self.Y

    @data.setter
    def data(self, _Y):
        self.Y = _Y
        self._adjust_format()

    def __getitem__(self, index):
        return self.Y[index] if self._valid else None


class DenseLabels(LabelsBase):
    """Class for dense labels (dense array is expected in __getitem__)

    Arguments:
    ----------
    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: np.ndarray or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the labels or not
        Useful in case of non binary labels
    """

    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)
        if normalize:
            self.normalize(norm='max')

    def __getitem__(self, index):
        return super().__getitem__(
            index).toarray().flatten().astype('float32')


class SparseLabels(LabelsBase):
    """Class for sparse labels (indices and weights are returned)

    Arguments:
    ----------

    data_dir: str
        data directory
    fname: str
        load data from this file
    Y: csr_matrix or None, optional, default=None
        data is already provided
    normalize: boolean, optional, default=False
        Normalize the labels or not
        Useful in case of non binary labels
    """

    def __init__(self, data_dir, fname, Y=None, normalize=False):
        super().__init__(data_dir, fname, Y)
        if normalize:
            self.normalize(norm='max')

    def __getitem__(self, index):
        y = self.Y[index].indices
        w = self.Y[index].data
        return y, w
